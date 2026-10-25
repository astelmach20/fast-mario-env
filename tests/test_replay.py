"""Frame-dedup replay buffer: sampled transitions vs explicitly stored stacks (needs torch)."""
import numpy as np
import pytest
from conftest import requires_rom, requires_torch, scaled

torch = pytest.importorskip("torch")


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@requires_rom
@requires_torch
@pytest.mark.parametrize("device", _devices())
def test_replay_exact(device):
    import fast_mario_env as fme
    from fast_mario_env.replay import FrameReplay

    n, steps, cap, n_step, gamma = 4, scaled(600), 200, 3, 0.9
    v = fme.FastVecEnv("1-1", n, seed=3, max_steps=40, sticky=0.25, noop_max=5)  # many short episodes
    rb = FrameReplay(cap, n, 4, device=device, seed=0)
    obs = v.reset()
    rng = np.random.default_rng(0)
    gt = {}
    for _ in range(steps):
        a = rng.integers(0, 7, n)
        o2, r, te, tr, inf = v.step(a)
        nxt = o2.copy()
        d = te | tr
        for i in np.flatnonzero(d):
            nxt[i] = inf[i]["final_obs"]
        rb.add(obs, nxt, a, r, d, tr & ~te)
        for i in range(n):
            gt[(i, int(rb.ptr[i]) - 2)] = (obs[i].copy(), int(a[i]), float(r[i]), nxt[i].copy(), bool(d[i]), bool(te[i]))
        obs = o2
    v.close()
    assert rb.ptr.min() > cap  # ring wrapped around
    bad = checked = 0
    for ns in (1, n_step):
        for _ in range(5):
            b = rb.sample(256, ns, gamma)
            for j in range(256):
                e, p, m = int(b.env[j]), int(b.pos[j]), int(b.steps[j])
                o, act = gt[(e, p)][:2]
                R, q, term, k = 0.0, p, False, 0
                for i in range(ns):
                    _, _, tr_, tn, td, tt = gt[(e, q)]
                    R += gamma ** i * tr_; k += 1; last = tn; term = tt
                    if td:
                        break
                    q += 1
                ok = (np.array_equal(b.obs[j].cpu().numpy(), o) and int(b.action[j]) == act and k == m
                      and np.array_equal(b.next_obs[j].cpu().numpy(), last) and bool(b.terminal[j]) == term
                      and abs(float(b.reward[j]) - R) < 1e-4
                      and abs(float(b.discount[j]) - (0 if term else gamma ** k)) < 1e-5)
                bad += not ok; checked += 1
    assert bad == 0, f"{bad}/{checked} sampled transitions differ"


@requires_rom
@requires_torch
def test_sb3_dqn_smoke():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import DQN
    import fast_mario_env as fme
    from fast_mario_env.replay import DedupReplayBuffer

    v = fme.sb3_vec_env("1-1", 2, seed=0, stack_in_cpp=True, sticky=0.25, noop_max=30)
    m = DQN("CnnPolicy", v, buffer_size=2_000, learning_starts=200, batch_size=32, train_freq=4,
            replay_buffer_class=DedupReplayBuffer, device="cpu", verbose=0, seed=0)
    m.learn(400)
    v.close()
    assert m.replay_buffer.size() > 0
