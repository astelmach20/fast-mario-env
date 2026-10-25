"""CUDA backend: CudaVecEnv (GPU) vs FastVecEnv (CPU, itself bit-exact vs nes-py), in lockstep.

Skipped without torch + a CUDA GPU. The first run compiles the kernels with NVRTC (about a minute;
cached afterwards) and boots each level on the GPU.
"""
import numpy as np
import pytest
from conftest import requires_cuda, requires_rom, scaled

import fast_mario_env as fme

pytestmark = [pytest.mark.cuda, requires_cuda, requires_rom]

CONFIGS = {
    "default": dict(),
    "sticky_noop": dict(sticky=0.25, noop_max=30),
    "short_timelimit": dict(max_steps=60, noop_max=5),
}
LEVELS = ["1-1", "1-2", "2-3", "3-4", "4-1", "5-2", "6-3", "7-4", "8-4"]


@pytest.mark.parametrize("config", list(CONFIGS))
def test_cuda_vs_cpu(config):
    import torch
    from fast_mario_env.cuda import CudaVecEnv

    kw = CONFIGS[config]
    levels = [lv for lv in LEVELS for _ in range(4)]
    n = len(levels)
    ref = fme.FastVecEnv(levels, n, seed=7, **kw)
    new = CudaVecEnv(levels, n, seed=7, **kw)
    rng = np.random.default_rng(130)
    assert np.array_equal(ref.reset(), new.reset().cpu().numpy())
    eps = 0
    for t in range(scaled(200)):
        a = rng.choice(7, n, p=[.05, .2, .25, .15, .25, .05, .05])
        o1, r1, te1, tr1, i1 = ref.step(a)
        o2, r2, te2, tr2, _ = new.step(torch.from_numpy(a).cuda())
        assert np.array_equal(o1, o2.cpu().numpy()), t
        assert np.array_equal(r1, r2.cpu().numpy()), t
        assert np.array_equal(te1, te2.cpu().numpy()) and np.array_equal(tr1, tr2.cpu().numpy()), t
        ram = new.ram().cpu().numpy()
        inf = new.info_t.cpu().numpy()
        for i in range(n):
            assert np.array_equal(ref.ram(i), ram[i]), (t, i)
            for j, k in enumerate(fme.INFO_KEYS):
                assert int(i1[i][k]) == int(inf[i, j]), (t, i, k)
        done = te1 | tr1
        if done.any():
            epr, epl, fo = new.ep_ret.cpu().numpy(), new.ep_len.cpu().numpy(), new.final_obs.cpu().numpy()
            for i in np.flatnonzero(done):
                eps += 1
                e = i1[i]["episode"]
                assert round(float(epr[i]), 6) == e["r"] and int(epl[i]) == e["l"], (t, i)
                assert np.array_equal(i1[i]["final_obs"], fo[i]), (t, i)
    assert new.bad_ops() == 0
    ref.close()
    if config == "short_timelimit":
        assert eps > 0


def test_cuda_v2_vs_v1():
    """The optimized kernel (v2) against the straightforward port (v1), all on the GPU."""
    import torch
    from fast_mario_env.cuda import CudaVecEnv

    levels = (list(fme.ALL_LEVELS) * 2)[:64]
    a = CudaVecEnv(levels, 64, seed=3, sticky=0.25, noop_max=30, kernel="v1")
    b = CudaVecEnv(levels, 64, seed=3, sticky=0.25, noop_max=30, kernel="v2")
    assert torch.equal(a.reset(), b.reset())
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    probs = torch.tensor([.05, .2, .25, .15, .25, .05, .05], device="cuda")
    for t in range(scaled(60)):
        act = torch.multinomial(probs, 64, replacement=True, generator=g)
        ra, rb = a.step(act), b.step(act)
        for x, y in zip(ra[:4], rb[:4]):
            assert torch.equal(x, y), t
        assert torch.equal(a.ram(), b.ram()), t
