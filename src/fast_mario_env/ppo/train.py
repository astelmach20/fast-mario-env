# SPDX-License-Identifier: GPL-3.0-or-later
"""Lean PPO for Super Mario Bros on fast-mario-env (CleanRL-style; this file + model.py + backends.py).

  * rollout storage on the GPU (uint8 obs); the C++ envs write into pinned host tensors; non_blocking H2D
  * env groups (--groups G): G independent C++ thread pools. While group g is stepped on the CPU the GPU
    runs inference for the other group(s), so the emulator does not wait for the policy
  * CUDA graphs: per-group policy forward + Gumbel-max sampling, PPO minibatch step (gather, fwd, bwd,
    grad clip, fused capturable Adam), GAE. Optional bf16 autocast (convs/fc; heads stay fp32)
  * --async-update 1: the PPO update of rollout k runs on a low-priority stream in a helper thread while
    rollout k+1 is collected with the pre-update weights; the new weights are swapped in as soon as the
    update finishes (policy lag <= 1 rollout; behaviour log-probs are stored so the ratio is exact)
  * env backend interface (backends.py): --env-backend cuda runs the GPU emulator (host=None device path)
  * logs: TB mario/{x_pos_mean100,flag_rate100,best_x_pos,episodes} + PPO stats, monitor.monitor.csv,
    progress.jsonl (periodic evals), result.json, model.pt checkpoint

  python -m fast_mario_env.ppo.train --preset sb3 --total-steps 3e6 --out runs/sb3eq_s0
  python -m fast_mario_env.ppo.train --preset sb3 --bench 20                    # throughput only
  python -m fast_mario_env.ppo.train --preset exp_fast --total-steps 5e6 --out runs/fast_s0   # EXPERIMENTAL
"""
import argparse
import collections
import copy
import json
import os
import sys
import threading
import time

import numpy as np
import torch

from .backends import CudaMarioBackend, FastMarioBackend
from .model import STAT_NAMES, NatureAC, ppo_loss

PRESETS = {
    # the SB3 PPO hyper-parameters this trainer was validated against (see docs/ppo.md)
    "sb3": dict(n_envs=8, groups=1, n_steps=512, batch=256, epochs=4, bf16=0, async_update=0),
    # EXPERIMENTAL throughput config: changes the algorithm (128 envs x 64 steps, batch 512, bf16, async update
    # with <=1-rollout policy lag). Not SB3-equivalent; see README "Experimental" section. Not a default.
    "exp_fast": dict(n_envs=128, groups=2, n_steps=64, batch=512, epochs=4, bf16=1, async_update=1),
}


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS), default="sb3",
                   help="sb3 (default): SB3-equivalent hyper-parameters; exp_fast: experimental, changes learning")
    p.add_argument("--level", default="1-1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--threads-per-group", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--adam-eps", type=float, default=1e-5)
    p.add_argument("--sticky", type=float, default=0.25)
    p.add_argument("--noop-max", type=int, default=30)
    p.add_argument("--max-ep-steps", type=int, default=2000)
    p.add_argument("--total-steps", type=float, default=3e6)
    p.add_argument("--eval-every", type=float, default=500_000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--target", type=float, default=None, help="stop once eval flag rate >= target")
    p.add_argument("--bf16", type=int, default=0)
    p.add_argument("--channels-last", type=int, default=1)
    p.add_argument("--graph", type=int, default=1, help="CUDA graphs for inference / update / GAE")
    p.add_argument("--async-update", type=int, default=0)
    p.add_argument("--device-env-test", type=int, default=0,
                   help="wrap the CPU env so it returns CUDA tensors with host=None (exercises the GPU-backend path)")
    p.add_argument("--out", default=None)
    p.add_argument("--bench", type=float, default=0, help="throughput mode: run N seconds (after warm-up), no logs")
    p.add_argument("--no-tb", action="store_true")
    p.add_argument("--check-nan", type=int, default=0)
    p.add_argument("--env-backend", choices=("cpu", "cuda"), default="cpu",
                   help="cpu: C++ FastVecEnv thread pool; cuda: CudaVecEnv on the GPU (bit-identical envs)")
    p.add_argument("--rom", default=None, help="path to the SMB ROM (default: $FAST_MARIO_ENV_ROM or gym-super-mario-bros)")
    argv = sys.argv[1:] if argv is None else argv
    a = p.parse_args(argv)
    given = {x.split("=")[0][2:].replace("-", "_") for x in argv if x.startswith("--")}
    if a.preset:
        for k, v in PRESETS[a.preset].items():
            if k not in given:
                setattr(a, k, v)
    a.total_steps, a.eval_every = int(a.total_steps), int(a.eval_every)
    assert a.n_envs % a.groups == 0 and (a.n_envs * a.n_steps) % a.batch == 0
    return a


class Storage:
    def __init__(self, T, N, dev):
        z = lambda *s, dt=torch.float32: torch.zeros(s, dtype=dt, device=dev)  # noqa: E731
        self.obs = z(T, N, 4, 84, 84, dt=torch.uint8)
        self.act = z(T, N, dt=torch.int64)
        self.logp, self.val, self.rew = z(T, N), z(T, N), z(T, N)
        self.done = z(T, N)  # done[t] = 1: obs[t] is the first obs of a new episode
        self.next_val, self.next_done = z(N), z(N)
        self.names = ("obs", "act", "logp", "val", "rew", "done", "next_val", "next_done")

    def copy_from(self, o):
        for k in self.names:
            getattr(self, k).copy_(getattr(o, k), non_blocking=True)


class Learner:
    def __init__(self, a, dev):
        self.a, self.dev = a, dev
        T, N = a.n_steps, a.n_envs
        self.net = NatureAC(7).to(dev)
        if a.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)
        self.act_net = self.net
        if a.async_update:  # separate acting copy, refreshed when an update finishes
            self.act_net = copy.deepcopy(self.net).requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=torch.tensor(a.lr, device=dev) if a.graph else a.lr,
                                    eps=a.adam_eps, fused=True, capturable=bool(a.graph))
        self.buf = Storage(T, N, dev)
        self.adv = torch.zeros(T, N, device=dev)
        self.ret = torch.zeros(T, N, device=dev)
        self.stats = torch.zeros(len(STAT_NAMES), device=dev)
        self.mb_idx = torch.zeros(a.batch, dtype=torch.int64, device=dev)
        self.ac = dict(device_type="cuda", dtype=torch.bfloat16, enabled=bool(a.bf16), cache_enabled=False)
        self.update_graph = self.gae_graph = None
        if a.graph:
            self._capture()

    def sync_act_net(self):
        if self.act_net is not self.net:
            with torch.no_grad():
                torch._foreach_copy_(list(self.act_net.parameters()), list(self.net.parameters()))

    def _gae(self):
        a, b = self.a, self.buf
        last = torch.zeros_like(b.next_val)
        for t in reversed(range(a.n_steps)):
            if t == a.n_steps - 1:
                nnt, nv = 1.0 - b.next_done, b.next_val
            else:
                nnt, nv = 1.0 - b.done[t + 1], b.val[t + 1]
            delta = b.rew[t] + a.gamma * nv * nnt - b.val[t]
            last = delta + (a.gamma * a.lam) * nnt * last
            self.adv[t] = last
        torch.add(self.adv, b.val, out=self.ret)

    def _mb_step(self):
        a, b, i = self.a, self.buf, self.mb_idx
        obs = b.obs.view(-1, 4, 84, 84)[i]
        with torch.autocast(**self.ac):
            loss, st = ppo_loss(self.net, obs, b.act.view(-1)[i], b.logp.view(-1)[i], self.adv.view(-1)[i],
                                self.ret.view(-1)[i], b.val.view(-1)[i], a.clip, a.ent, a.vf,
                                channels_last=bool(a.channels_last))
        self.opt.zero_grad(set_to_none=not self.a.graph)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), a.max_grad_norm, foreach=True)
        ok = torch.isfinite(gn)
        for p in self.net.parameters():  # skip (zero) a non-finite minibatch instead of poisoning the weights
            p.grad.copy_(torch.where(ok, p.grad, torch.zeros_like(p.grad)))
        self.opt.step()
        self.stats[:-1].add_(st)
        self.stats[-1].add_((~ok).float())

    def _capture(self):
        """Capture GAE + minibatch-step graphs once (on dummy data), then restore params / Adam state.
        NB: every graph is captured on its *own* stream. cuBLAS workspaces are cached per (handle, stream); graphs
        captured on the shared default capture stream share one workspace and corrupt each other (NaNs) when
        the update graph and an inference graph replay concurrently in --async-update mode."""
        snap = [p.detach().clone() for p in self.net.parameters()]
        self.mb_idx.copy_(torch.randint(0, self.a.n_steps * self.a.n_envs, (self.a.batch,), device=self.dev))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._gae(); self._mb_step()
        torch.cuda.current_stream().wait_stream(s)
        self.gae_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.gae_graph, stream=s):
            self._gae()
        self.opt.zero_grad(set_to_none=False)
        self.update_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.update_graph, stream=s):
            self._mb_step()
        with torch.no_grad():
            for p, q in zip(self.net.parameters(), snap):
                p.copy_(q)
            for st in self.opt.state.values():
                for k in ("exp_avg", "exp_avg_sq", "step"):
                    st[k].zero_()
        self.sync_act_net()
        torch.cuda.synchronize()

    def gae(self):
        self.gae_graph.replay() if self.gae_graph is not None else self._gae()

    def update(self, lr=None):
        """GAE + n_epochs x shuffled minibatches over self.buf. Returns mean stats tensor (no sync)."""
        a = self.a
        if lr is not None:
            for g in self.opt.param_groups:
                if torch.is_tensor(g["lr"]):
                    g["lr"].fill_(lr)
                else:
                    g["lr"] = lr
        self.gae()
        TN = a.n_steps * a.n_envs
        nmb = TN // a.batch
        self.stats.zero_()
        for _ in range(a.epochs):
            perm = torch.randperm(TN, device=self.dev)
            for k in range(nmb):
                self.mb_idx.copy_(perm[k * a.batch:(k + 1) * a.batch])
                self.update_graph.replay() if self.update_graph is not None else self._mb_step()
        return self.stats / (a.epochs * nmb)


class Policy:
    """Per-group inference on static buffers (CUDA graph)."""

    def __init__(self, learner, n, dev, graph):
        self.L = learner
        self.inp = torch.zeros((n, 4, 84, 84), dtype=torch.uint8, device=dev)
        self.a = torch.zeros(n, dtype=torch.int64, device=dev)
        self.lp, self.v = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        self.graph = None
        if graph:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._run()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=s):
                self._run()

    def _run(self):
        with torch.autocast(**self.L.ac):
            act, lp, v = self.L.act_net.act(self.inp, channels_last=bool(self.L.a.channels_last))
        self.a.copy_(act); self.lp.copy_(lp); self.v.copy_(v)

    def __call__(self):
        self.graph.replay() if self.graph is not None else self._run()


class DeviceEnv:
    """Test adapter: makes a CPU backend look like a GPU emulator (CUDA tensors out, host=None)."""

    def __init__(self, env, dev):
        self.e, self.dev, self.n_envs, self.device = env, dev, env.n_envs, dev

    def reset(self):
        return self.e.reset().to(self.dev)

    def step_async(self, actions):
        self.e.step_async(actions)

    def step_wait(self):
        o = self.e.step_wait()
        c = lambda x: x.to(self.dev).clone()  # noqa: E731
        o2 = type(o)(*[c(getattr(o, f)) for f in ("obs", "reward", "terminated", "truncated", "ep_return", "ep_len",
                                                  "x_pos", "flag_get")], host=None)
        return o2

    def final_obs(self, idx):
        return self.e.final_obs(idx.cpu()).to(self.dev)

    def close(self):
        self.e.close()


@torch.no_grad()
def evaluate(net, a, episodes, seed, dev):
    """One stochastic episode in each of `episodes` parallel envs (same sticky/noop perturbations as training)."""
    env = FastMarioBackend(a.level, episodes, seed=seed, sticky=a.sticky, noop_max=a.noop_max, max_steps=a.max_ep_steps,
                           rom_path=a.rom)
    obs, res = env.reset(), {}
    ac = dict(device_type="cuda", dtype=torch.bfloat16, enabled=bool(a.bf16), cache_enabled=False)
    while len(res) < episodes:
        with torch.autocast(**ac):
            act, _, _ = net.act(obs.to(dev), channels_last=bool(a.channels_last))
        env.step_async(act.cpu())
        h = env.step_wait().host
        for i in np.flatnonzero(h["term"] | h["trunc"]):
            if i not in res:
                res[i] = (float(h["info"][i, 0]), bool(h["info"][i, 1]))
    env.close()
    xs = [r[0] for r in res.values()]
    return dict(flag_rate=float(np.mean([r[1] for r in res.values()])), x_mean=float(np.mean(xs)), x_max=float(np.max(xs)))


class EpisodeLog:
    def __init__(self, out, t0, no_tb):
        self.recent = collections.deque(maxlen=100)
        self.best_x, self.n, self.t0 = 0.0, 0, t0
        self.mon = self.tb = None
        if out:
            self.mon = open(os.path.join(out, "monitor.monitor.csv"), "w")
            self.mon.write("#%s\n" % json.dumps({"t_start": time.time(), "env_id": "None"}))
            self.mon.write("r,l,t,x_pos,flag_get\n")
            if not no_tb:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(out, "PPO_1"))

    def add(self, r, l, x, f):
        self.n += 1
        self.recent.append((x, f, r, l))
        self.best_x = max(self.best_x, x)
        if self.mon:
            self.mon.write(f"{round(r, 6)},{l},{round(time.time() - self.t0, 6)},{x},{bool(f)}\n")

    def scalars(self):
        if not self.recent:
            return {}
        r = np.array(self.recent, dtype=np.float64)
        return {"mario/x_pos_mean100": r[:, 0].mean(), "mario/flag_rate100": r[:, 1].mean(),
                "mario/best_x_pos": self.best_x, "mario/episodes": self.n,
                "rollout/ep_rew_mean": r[:, 2].mean(), "rollout/ep_len_mean": r[:, 3].mean()}

    def write(self, d, step):
        if self.tb:
            for k, v in d.items():
                self.tb.add_scalar(k, v, step)
        if self.mon:
            self.mon.flush()


def other_gpu_procs():
    """pids of other processes holding a CUDA context (to flag contaminated wall-clock numbers)."""
    import subprocess
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=10).stdout
        return [int(x) for x in o.split() if x.strip().isdigit() and int(x) != os.getpid()]
    except Exception:
        return []


def main(argv=None):
    a = parse(argv)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    out = None if a.bench else a.out
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)

    # collection runs on a high-priority stream so inference is not queued behind (async) update kernels
    cstream = torch.cuda.Stream(priority=-1)
    torch.cuda.set_stream(cstream)
    ustream = torch.cuda.Stream(priority=0) if a.async_update else None

    G, ng = a.groups, a.n_envs // a.groups
    Backend = CudaMarioBackend if a.env_backend == "cuda" else FastMarioBackend
    envs = [Backend(a.level, ng, seed=a.seed, sticky=a.sticky, noop_max=a.noop_max, max_steps=a.max_ep_steps,
                    n_threads=min(a.threads_per_group, ng), env_offset=g * ng, rom_path=a.rom) for g in range(G)]
    if a.device_env_test:
        envs = [DeviceEnv(e, dev) for e in envs]
    L = Learner(a, dev)
    R = L.buf if not a.async_update else Storage(a.n_steps, a.n_envs, dev)  # collector's buffers
    pols = [Policy(L, ng, dev, a.graph) for _ in range(G)]
    sls = [slice(g * ng, (g + 1) * ng) for g in range(G)]
    act_pin = [torch.zeros(ng, dtype=torch.int64, pin_memory=True) for _ in range(G)]
    evs = [torch.cuda.Event() for _ in range(G)]
    dev_eps = []  # device-backend episode records (per step), flushed once per rollout
    for g in range(G):
        pols[g].inp.copy_(envs[g].reset(), non_blocking=True)
    torch.cuda.synchronize()

    t0 = time.time()
    log = EpisodeLog(out, t0, a.no_tb)
    tm = collections.defaultdict(float)
    steps, it, next_eval, eval_s = 0, 0, a.eval_every, 0.0
    prog, status = [], "max_steps"
    gpu_busy_evals = []  # eval points at which another process was using the GPU (wall-clock contaminated)
    upd = dict(thread=None, stats=None)

    def launch(g, t):
        p, sl = pols[g], sls[g]
        R.obs[t, sl].copy_(p.inp)
        p()
        R.act[t, sl].copy_(p.a); R.logp[t, sl].copy_(p.lp); R.val[t, sl].copy_(p.v)
        e = envs[g]
        if e.device.type == "cpu":  # pinned host buffers: wait for the action D2H, then step
            e.act.copy_(p.a, non_blocking=True)
            evs[g].record()
            t1 = time.perf_counter(); evs[g].synchronize(); tm["gpu_wait"] += time.perf_counter() - t1
            e.step_async(e.act)
        else:
            e.step_async(p.a)

    def receive(g, t):
        p, sl, e = pols[g], sls[g], envs[g]
        t1 = time.perf_counter()
        o = e.step_wait()
        tm["env_wait"] += time.perf_counter() - t1
        p.inp.copy_(o.obs, non_blocking=True)
        R.rew[t, sl].copy_(o.reward, non_blocking=True)
        dst = R.done[t + 1, sl] if t + 1 < a.n_steps else R.next_done[sl]
        term = o.terminated.to(dev, non_blocking=True)
        trunc = o.truncated.to(dev, non_blocking=True)
        torch.maximum(term, trunc, out=term)
        dst.copy_(term)
        h = o.host
        if h is not None:  # CPU backend: cheap host-side bookkeeping on the pinned numpy views
            d = h["term"] | h["trunc"]
            if d.any():
                idx = np.flatnonzero(d)
                tr = idx[(h["trunc"][idx] > 0) & (h["term"][idx] == 0)]
                if len(tr):  # SB3 timeout bootstrap: r += gamma * V(terminal obs)
                    ti = torch.from_numpy(tr)
                    fo = e.final_obs(ti).to(dev)
                    with torch.no_grad(), torch.autocast(**L.ac):
                        _, v = L.act_net(fo, bool(a.channels_last))
                    R.rew[t, sl.start + ti.to(dev)] += a.gamma * v
                inf, epr, epl = h["info"], h["epr"], h["epl"]
                for i in idx:
                    log.add(float(epr[i]), int(epl[i]), int(inf[i, 0]), bool(inf[i, 1]))
        else:  # device backend: everything stays on the GPU
            tmask = (o.truncated > 0) & (o.terminated == 0)
            if bool(tmask.any()):  # rare (time limit); one tiny sync
                ti = tmask.nonzero().squeeze(-1)
                with torch.no_grad(), torch.autocast(**L.ac):
                    _, v = L.act_net(e.final_obs(ti), bool(a.channels_last))
                R.rew[t, sl.start + ti] += a.gamma * v
            dev_eps.append(torch.stack([(o.terminated > 0) | (o.truncated > 0), o.ep_return.float(), o.ep_len.float(),
                                        o.x_pos.float(), o.flag_get.float()]).float())

    def flush_dev_eps():
        if dev_eps:
            x = torch.stack(dev_eps).cpu().numpy()  # (steps, 5, n)
            dev_eps.clear()
            for s in range(x.shape[0]):
                for i in np.flatnonzero(x[s, 0]):
                    log.add(float(x[s, 1, i]), int(x[s, 2, i]), int(x[s, 3, i]), bool(x[s, 4, i]))

    def run_update(lr):
        with torch.cuda.stream(ustream):
            st = L.update(lr)
            upd["stats"] = st.tolist()  # synchronises ustream only

    def finish_update():
        if upd["thread"] is not None:
            t1 = time.perf_counter()
            upd["thread"].join()
            tm["update_wait"] += time.perf_counter() - t1
            upd["thread"] = None
            L.sync_act_net()  # on cstream; the update stream has finished (tolist synced it)

    n_iters = max(1, a.total_steps // (a.n_envs * a.n_steps))
    warm = 2 if a.bench else 0
    while steps < a.total_steps:
        it += 1
        if it == warm + 1 and a.bench:  # restart clocks after warm-up (cudnn autotune etc.)
            torch.cuda.synchronize(); t0 = time.time(); steps = 0; tm.clear()
        lr = a.lr * (1.0 - (it - 1) / n_iters) if a.anneal_lr else None
        tc = time.perf_counter()
        for g in range(G):
            launch(g, 0)
        for t in range(a.n_steps):
            for g in range(G):
                receive(g, t)
                if t + 1 < a.n_steps:
                    launch(g, t + 1)
            if upd["thread"] is not None and not upd["thread"].is_alive():
                finish_update()  # swap in the new weights mid-rollout
        for g in range(G):  # bootstrap value of the final obs
            pols[g]()
            R.next_val[sls[g]].copy_(pols[g].v)
        flush_dev_eps()
        if a.check_nan:
            bad = {k: int((~torch.isfinite(getattr(R, k).float())).sum()) for k in ("logp", "val", "rew", "next_val")}
            bad["act_net"] = int(sum((~torch.isfinite(q)).sum() for q in L.act_net.parameters()))
            bad["net"] = int(sum((~torch.isfinite(q)).sum() for q in L.net.parameters()))
            if any(bad.values()):
                print("NONFINITE it", it, bad, "act range", int(R.act.min()), int(R.act.max()), flush=True)
                fl = (~torch.isfinite(R.logp)).nonzero()[:5].tolist(); print("  at (t,env)", fl, flush=True)
        tm["collect"] += time.perf_counter() - tc
        steps += a.n_envs * a.n_steps
        tu = time.perf_counter()
        if a.async_update:
            finish_update()
            stats = upd["stats"]
            L.buf.copy_from(R)  # on cstream
            ustream.wait_stream(cstream)
            upd["thread"] = threading.Thread(target=run_update, args=(lr,), daemon=True)
            upd["thread"].start()
        else:
            stats = L.update(lr).tolist()
        tm["update"] += time.perf_counter() - tu

        if a.bench:
            if it > warm and time.time() - t0 > a.bench:
                break
            continue
        wall = time.time() - t0
        sc = log.scalars()
        sc["time/fps"] = steps / max(1e-9, wall - eval_s)
        if stats is not None:
            for k, v in zip(STAT_NAMES, stats):
                sc["train/" + k] = v
        log.write(sc, steps)
        if it % 10 == 1:
            print(f"it {it} steps {steps} fps {sc['time/fps']:.0f} " +
                  " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in sc.items() if k.startswith(("mario/", "train/approx", "train/nonf", "train/entropy"))), flush=True)
        if steps >= next_eval or steps >= a.total_steps:
            next_eval += a.eval_every
            finish_update()
            busy = other_gpu_procs()
            if busy:
                gpu_busy_evals.append(steps)
            te = time.time()
            r = evaluate(L.net, a, a.eval_episodes, a.seed + steps, dev)
            eval_s += time.time() - te
            rec = dict(steps=steps, wall_s=round(time.time() - t0, 1), train_wall_s=round(time.time() - t0 - eval_s, 1),
                       **r, train_flag_rate100=sc.get("mario/flag_rate100"), train_x_mean100=sc.get("mario/x_pos_mean100"))
            prog.append(rec)
            if out:
                with open(os.path.join(out, "progress.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            print("EVAL", json.dumps(rec), flush=True)
            if log.tb:
                for k in ("flag_rate", "x_mean"):
                    log.tb.add_scalar("eval/" + k, r[k], steps)
            if out:
                torch.save(dict(net=L.net.state_dict(), opt=L.opt.state_dict(), steps=steps, args=vars(a)),
                           os.path.join(out, "model.pt"))
            if a.target is not None and r["flag_rate"] >= a.target:
                status = "cleared"
                break
    finish_update()
    torch.cuda.synchronize()
    wall = time.time() - t0
    for e in envs:
        e.close()
    res = dict(level=a.level, status=status, steps=steps, wall_s=round(wall, 1), train_wall_s=round(wall - eval_s, 1),
               fps=round(steps / (wall - eval_s)), final_eval=prog[-1] if prog else None, n_envs=a.n_envs,
               sticky=a.sticky, noop_max=a.noop_max, timers_s={k: round(v, 3) for k, v in tm.items()},
               config=vars(a), load1=round(os.getloadavg()[0], 2), other_gpu_procs_at_steps=gpu_busy_evals)
    if out:
        json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
        if log.tb:
            log.tb.close()
    print("DONE", json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    main()
