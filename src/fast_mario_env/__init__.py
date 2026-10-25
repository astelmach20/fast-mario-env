"""fast-mario-env: a bit-exact, batched, much faster Super Mario Bros environment for RL.

    from fast_mario_env import FastVecEnv, sb3_vec_env
    from fast_mario_env.cuda import CudaVecEnv          # optional, needs torch + an NVIDIA GPU
    from fast_mario_env.replay import FrameReplay       # optional, needs torch
"""
from .env import (INFO_FIELDS, INFO_KEYS, N_ACTIONS, OBS_HW, SIMPLE_MOVEMENT, FastVecEnv, MonitorCSV, RawNES,
                  gray_lut, palette_rgb, resize_area, rng_state)
from .rom import ALL_LEVELS, ROM_ENV_VAR, RomNotFoundError, decode_level, find_rom, load_rom
from .sb3 import sb3_vec_env

__version__ = "0.1.0"

__all__ = ["FastVecEnv", "sb3_vec_env", "RawNES", "MonitorCSV", "palette_rgb", "gray_lut", "rng_state",
           "resize_area", "find_rom", "load_rom", "decode_level", "RomNotFoundError", "ROM_ENV_VAR", "ALL_LEVELS",
           "OBS_HW", "N_ACTIONS", "SIMPLE_MOVEMENT", "INFO_FIELDS", "INFO_KEYS", "__version__"]
