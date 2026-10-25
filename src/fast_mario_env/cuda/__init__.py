"""Optional CUDA backend: thousands of bit-exact Super Mario Bros envs on one NVIDIA GPU.

Needs torch with CUDA (``pip install fast-mario-env[cuda]``). Kernels are compiled at first use with
NVRTC from torch's CUDA wheels (no nvcc / host compiler needed) and cached on disk.

    from fast_mario_env.cuda import CudaVecEnv
"""
from .vec_env import INFO_FIELDS, CudaVecEnv, get_module

__all__ = ["CudaVecEnv", "INFO_FIELDS", "get_module"]
