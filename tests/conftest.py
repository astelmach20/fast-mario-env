import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # reference_env

# FME_TEST_SCALE multiplies the number of frames / steps of the fidelity tests (default 1 = CI size).
SCALE = max(1, int(os.environ.get("FME_TEST_SCALE", "1")))


def scaled(n):
    return n * SCALE


def _rom_available():
    try:
        from fast_mario_env import find_rom
        find_rom()
        return True
    except Exception:
        return False


def _have(*mods):
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in mods)


HAVE_ROM = _rom_available()
HAVE_REF = _have("nes_py", "gym_super_mario_bros", "cv2")


def _have_cuda():
    if not _have("torch"):
        return False
    import torch
    return torch.cuda.is_available()


requires_rom = pytest.mark.skipif(not HAVE_ROM, reason="no SMB ROM (install gym-super-mario-bros or set FAST_MARIO_ENV_ROM)")
requires_ref = pytest.mark.skipif(not (HAVE_REF and HAVE_ROM), reason="reference stack (nes-py, gym-super-mario-bros, cv2) not installed")
requires_torch = pytest.mark.skipif(not _have("torch"), reason="torch not installed")
requires_cuda = pytest.mark.skipif(not _have_cuda(), reason="no CUDA GPU / torch")
