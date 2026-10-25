"""API, ROM handling and determinism tests (no reference stack needed; a ROM is needed for most)."""
import numpy as np
import pytest
from conftest import requires_rom

import fast_mario_env as fme
from fast_mario_env import rom


def test_decode_level():
    assert fme.decode_level("1-1") == (1, 1, 1)
    assert fme.decode_level("1-2") == (1, 2, 3)
    assert fme.decode_level((4, 4)) == (4, 4, 5)
    assert fme.decode_level("3-2") == (3, 2, 2)
    assert fme.decode_level("8-4") == (8, 4, 4)
    for bad in ("0-1", "9-1", "1-5", "x"):
        with pytest.raises(ValueError):
            fme.decode_level(bad)


def test_decode_level_matches_gym_super_mario_bros():
    pytest.importorskip("gym_super_mario_bros")
    from gym_super_mario_bros._roms import decode_target
    for lv in fme.ALL_LEVELS:
        w, s = (int(x) for x in lv.split("-"))
        assert fme.decode_level(lv) == decode_target((w, s), False)


def test_missing_rom_message(tmp_path, monkeypatch):
    monkeypatch.setenv(fme.ROM_ENV_VAR, str(tmp_path / "nope.nes"))
    with pytest.raises(fme.RomNotFoundError, match="does not ship a ROM"):
        fme.find_rom()
    with pytest.raises(fme.RomNotFoundError):
        fme.FastVecEnv("1-1", 1)


def test_bad_rom_rejected(tmp_path):
    p = tmp_path / "bad.nes"
    p.write_bytes(b"not a rom" * 100)
    with pytest.raises(ValueError, match="iNES"):
        fme.load_rom(p)


@requires_rom
def test_rom_path_and_env_var(monkeypatch):
    path = fme.find_rom()
    monkeypatch.setenv(fme.ROM_ENV_VAR, path)
    assert fme.find_rom() == path
    monkeypatch.delenv(fme.ROM_ENV_VAR)
    with fme.FastVecEnv("1-1", 1, rom_path=path) as v:
        assert v.reset().shape == (1, 4, 84, 84)


@requires_rom
def test_shapes_and_infos():
    v = fme.FastVecEnv("1-1", 3, seed=0, max_steps=20, n_threads=2)
    obs = v.reset()
    assert obs.shape == (3, 4, 84, 84) and obs.dtype == np.uint8
    assert (obs[:, :3] == 0).all() and obs[:, 3].any()  # zero-padded stack after reset
    ended = 0
    for t in range(25):
        obs, r, te, tr, infos = v.step(np.full(3, 1))
        assert r.dtype == np.float32 and te.dtype == bool and tr.dtype == bool and len(infos) == 3
        assert set(fme.INFO_KEYS) <= set(infos[0])
        for i in np.flatnonzero(te | tr):
            ended += 1
            assert infos[i]["final_obs"].shape == (4, 84, 84)
            assert infos[i]["episode"]["l"] == 20 and infos[i]["TimeLimit.truncated"]
    assert ended == 3
    assert v.bad_ops() == 0 and v.frames > 0
    v.close()


@requires_rom
def test_determinism_and_threads():
    """Results do not depend on the thread count; the same seed gives the same trajectory."""
    acts = np.random.default_rng(0).integers(0, 7, (60, 4))
    outs = []
    for threads in (0, 1, 3):
        v = fme.FastVecEnv(["1-1", "1-2", "4-1", "8-4"], 4, seed=5, sticky=0.25, noop_max=10, n_threads=threads)
        o = [v.reset()]
        for a in acts:
            o.append(v.step(a)[0])
        outs.append(np.stack(o))
        v.close()
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2])


@requires_rom
def test_env_offset_matches_single_env():
    acts = np.random.default_rng(1).integers(0, 7, (40, 4))
    a = fme.FastVecEnv("1-1", 4, seed=2, sticky=0.25, noop_max=10)
    b = fme.FastVecEnv("1-1", 2, seed=2, sticky=0.25, noop_max=10, env_offset=2)
    oa, ob = a.reset(), b.reset()
    assert np.array_equal(oa[2:], ob)
    for x in acts:
        oa, ob = a.step(x)[0], b.step(x[2:])[0]
        assert np.array_equal(oa[2:], ob)


@requires_rom
def test_sb3_adapter():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
    venv = VecFrameStack(VecMonitor(fme.sb3_vec_env("1-1", 2, seed=0)), 4)
    o1 = venv.reset()
    fast = fme.sb3_vec_env("1-1", 2, seed=0, stack_in_cpp=True)
    o2 = fast.reset()
    assert o1.shape == (2, 84, 84, 4) and o2.shape == (2, 4, 84, 84)
    assert np.array_equal(o1.transpose(0, 3, 1, 2), o2)
    for t in range(30):
        a = np.array([t % 7, (t * 3) % 7])
        o1, _, d1, _ = venv.step(a)
        o2, _, d2, _ = fast.step(a)
        assert np.array_equal(o1.transpose(0, 3, 1, 2), o2) and np.array_equal(d1, d2)
    venv.close(); fast.close()


def test_rom_file_table():
    assert rom.ROM_FILES["vanilla"] == "super-mario-bros.nes"
