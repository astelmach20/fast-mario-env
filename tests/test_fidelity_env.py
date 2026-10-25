"""Env-level fidelity: FastVecEnv vs the reference nes-py stack (tests/reference_env.py), step for step.

Same seeds and the same random action sequence; every step must give identical stacked obs, rewards,
terminated, truncated, info values, episode stats, final_obs, TimeLimit.truncated, and the full 2 KB
RAM of every env. Configurations cover sticky actions, no-op starts, max-pooling and the time limit.
Set FME_TEST_POLICY=/path/to/sb3_ppo.zip to drive the actions with a trained SB3 agent instead (which
reaches flags, pipes and the time-limit paths that random play rarely does).
"""
import os

import numpy as np
import pytest
from conftest import requires_ref, scaled

import fast_mario_env as fme

CONFIGS = {
    "default": dict(),
    "sticky_noop": dict(sticky=0.25, noop_max=30),
    "maxpool_sticky": dict(maxpool=True, sticky=0.25),
    "short_timelimit": dict(max_steps=60, noop_max=5),
}
ACTION_P = [.05, .2, .25, .15, .25, .05, .05]  # biased to the right so episodes progress, die and time out


def _policy():
    path = os.environ.get("FME_TEST_POLICY")
    if not path:
        return None
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu")


def run(kw, steps, n, level, seed=7, policy=None):
    from reference_env import RefVecEnv
    ref = RefVecEnv(level, n, seed=seed, **kw)
    new = fme.FastVecEnv(level, n, seed=seed, n_threads=min(n, 4), **kw)
    rng = np.random.default_rng(123 + seed)
    mism = []

    def chk(cond, what, t):
        if not cond and len(mism) < 5:
            mism.append(dict(step=t, what=what))

    o1, o2 = ref.reset(), new.reset()
    chk(np.array_equal(o1, o2), "reset_obs", -1)
    eps = 0
    for t in range(steps):
        if policy is not None:
            a, _ = policy.predict(o1.transpose(0, 2, 3, 1), deterministic=False)
        else:
            a = rng.choice(7, n, p=ACTION_P)
        o1, r1, te1, tr1, i1 = ref.step(a)
        o2, r2, te2, tr2, i2 = new.step(a)
        chk(np.array_equal(o1, o2), "obs", t)
        chk(np.array_equal(r1, r2), "reward", t)
        chk(np.array_equal(te1, te2), "terminated", t)
        chk(np.array_equal(tr1, tr2), "truncated", t)
        for i in range(n):
            chk(np.array_equal(ref.ram(i), new.ram(i)), f"ram[{i}]", t)
            a_, b_ = i1[i], i2[i]
            for k in fme.INFO_KEYS:
                chk(int(a_[k]) == int(b_[k]), f"info.{k}[{i}]", t)
            chk(("episode" in a_) == ("episode" in b_), f"episode_end[{i}]", t)
            if "episode" in a_ and "episode" in b_:
                eps += 1
                ea, eb = a_["episode"], b_["episode"]
                chk(all(ea[k] == eb[k] for k in ("r", "l", "x_pos", "flag_get")), f"episode[{i}]", t)
                chk(np.array_equal(a_["final_obs"], b_["final_obs"]), f"final_obs[{i}]", t)
                chk(a_["TimeLimit.truncated"] == b_["TimeLimit.truncated"], f"TimeLimit.truncated[{i}]", t)
        if mism:
            break
    bad_ops = new.bad_ops()
    new.close()
    return dict(steps=t + 1, episodes=eps, mismatches=mism, bad_ops=bad_ops)


@requires_ref
@pytest.mark.parametrize("level", ["1-1", "4-1"])
@pytest.mark.parametrize("config", list(CONFIGS))
def test_env_vs_reference(config, level):
    steps = scaled(400 if config == "short_timelimit" else 250)
    r = run(CONFIGS[config], steps, 2, level, policy=_policy())
    print(config, level, r)
    assert not r["mismatches"], r
    assert r["bad_ops"] == 0
    if config == "short_timelimit":
        assert r["episodes"] >= 2


@requires_ref
def test_env_mixed_levels():
    """A list of levels (one per env) behaves like separate single-level envs."""
    r = run(dict(sticky=0.25, noop_max=5), scaled(150), 3, ["1-2", "5-3", "8-4"])
    assert not r["mismatches"], r
