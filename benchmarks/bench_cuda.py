"""Env-only throughput of CudaVecEnv: random actions generated on the GPU, steps/s vs n_envs.

    python benchmarks/bench_cuda.py --n 1024,4096,16384,32768 --level all

--level all mixes all 32 levels (env i plays level i % 32). Prints one JSON line per n_envs,
including GPU utilisation before/after (from nvidia-smi) so that contention is visible.
"""
import argparse
import json
import subprocess
import time

import torch

import fast_mario_env as fme
from fast_mario_env.cuda import CudaVecEnv


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


p = argparse.ArgumentParser()
p.add_argument("--n", default="1024,4096,16384")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--warm", type=int, default=5)
p.add_argument("--level", default="all")
p.add_argument("--sticky", type=float, default=0.25)
p.add_argument("--noop", type=int, default=30)
p.add_argument("--actions", default="random", choices=["random", "same"],
               help="same: all envs get the same action (less warp divergence)")
a = p.parse_args()
for n in [int(x) for x in a.n.split(",")]:
    levels = [fme.ALL_LEVELS[i % 32] for i in range(n)] if a.level == "all" else a.level
    t0 = time.time()
    env = CudaVecEnv(levels, n, seed=0, sticky=a.sticky, noop_max=a.noop)
    tinit = time.time() - t0
    env.reset()
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    probs = torch.tensor([.05, .2, .25, .15, .25, .05, .05], device="cuda")
    acts = [torch.multinomial(probs, n, replacement=True, generator=g) for _ in range(8)]
    if a.actions == "same":
        acts = [x[:1].expand(n).contiguous() for x in acts]
    for i in range(a.warm):
        env.step(acts[i % 8])
    torch.cuda.synchronize()
    smi0, f0, t0 = smi(), env.frames, time.time()
    for i in range(a.steps):
        env.step(acts[i % 8])
    torch.cuda.synchronize()
    dt = time.time() - t0
    r = dict(n_envs=n, level=a.level, actions=a.actions, steps_per_s=round(n * a.steps / dt),
             frames_per_s=round((env.frames - f0) / dt), ms_per_step=round(1000 * dt / a.steps, 2),
             bytes_per_env=round(env.bytes_per_env()), init_s=round(tinit, 1), smi_before=smi0, smi_after=smi())
    print(json.dumps(r), flush=True)
    del env
    torch.cuda.empty_cache()
