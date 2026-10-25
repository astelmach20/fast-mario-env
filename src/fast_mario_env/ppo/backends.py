"""Env backends for the PPO trainer.

Interface (duck-typed; see `EnvBackend`):
    n_envs, device                  # device where step outputs live ('cpu' pinned, or 'cuda')
    reset() -> obs uint8 (N,4,84,84) tensor
    step_async(actions)             # int64 (N,) tensor (cuda or cpu)
    step_wait() -> StepOut          # tensors; no python per-env work
    final_obs(idx) -> uint8 (len(idx),4,84,84) pre-reset obs (for timeout bootstrap)

StepOut (length-N tensors): obs, reward f32, terminated, truncated (uint8/bool), and episode stats
valid where terminated|truncated: ep_return, ep_len, x_pos, flag_get. `host`: optional dict of numpy
views (term, trunc, info, epr, epl) for cheap host-side bookkeeping. A GPU emulator returns CUDA
tensors and host=None; the trainer then keeps done flags / episode stats on the device and moves
them to the host once per rollout (no per-step host copies).

FastMarioBackend: the C++ FastVecEnv (one thread pool per instance). The C++ side writes
obs/reward/done/info straight into *pinned* torch tensors (via FastVecEnv.set_buffers), so the
H2D is a single non_blocking DMA per step and step_wait does no per-env python work.
CudaMarioBackend: fast_mario_env.cuda.CudaVecEnv behind the same interface (device path).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..env import INFO_FIELDS, FastVecEnv


def _tids():
    return {int(t) for t in os.listdir("/proc/self/task")} if os.path.isdir("/proc/self/task") else set()


@dataclass
class StepOut:
    obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    ep_return: torch.Tensor
    ep_len: torch.Tensor
    x_pos: torch.Tensor
    flag_get: torch.Tensor
    host: Optional[dict] = None


class EnvBackend:
    n_envs: int
    device: torch.device

    def reset(self) -> torch.Tensor: ...
    def step_async(self, actions: torch.Tensor): ...
    def step_wait(self) -> StepOut: ...
    def final_obs(self, idx) -> torch.Tensor: ...
    def close(self): ...


class FastMarioBackend(EnvBackend):
    """CPU emulator. Returned tensors are pinned host buffers overwritten by the next step: the caller
    must make sure its non_blocking H2D copies completed before the next step_async (the trainer
    synchronises on the action D2H, queued later on the same stream).

    env_offset: env i of this instance gets RNG stream seed*1000 + env_offset + i, so G instances of N/G
    envs use exactly the streams of one FastVecEnv(seed, N).
    pin_cpus: optional list of logical CPUs to pin the C++ pool threads to (Linux only)."""

    def __init__(self, level="1-1", n_envs=8, seed=0, sticky=0.25, noop_max=30, max_steps=2000,
                 n_threads=None, frame_stack=4, env_offset=0, pin_cpus=None, rom_path=None):
        self.n_envs, self.k = n_envs, frame_stack
        self.device = torch.device("cpu")
        tids0 = _tids()
        self.v = FastVecEnv(level, n_envs, seed=seed, frame_stack=frame_stack, sticky=sticky, noop_max=noop_max,
                            max_steps=max_steps, n_threads=n_threads, copy_obs=False, env_offset=env_offset,
                            rom_path=rom_path)
        self.pool_tids = sorted(_tids() - tids0)
        if pin_cpus:
            for j, tid in enumerate(self.pool_tids):
                try:
                    os.sched_setaffinity(tid, {pin_cpus[j % len(pin_cpus)]})
                except (OSError, AttributeError):
                    pass
        n, k = n_envs, frame_stack
        pin = dict(pin_memory=torch.cuda.is_available())
        self.obs = torch.zeros((n, k, 84, 84), dtype=torch.uint8, **pin)
        self.fin = torch.zeros((n, k, 84, 84), dtype=torch.uint8, **pin)
        self.rew = torch.zeros(n, dtype=torch.float32, **pin)
        self.term = torch.zeros(n, dtype=torch.uint8, **pin)
        self.trunc = torch.zeros(n, dtype=torch.uint8, **pin)
        self.info = torch.zeros((n, len(INFO_FIELDS)), dtype=torch.int64, **pin)
        self.epr = torch.zeros(n, dtype=torch.float64, **pin)
        self.epl = torch.zeros(n, dtype=torch.int64, **pin)
        self.act = torch.zeros(n, dtype=torch.int64, **pin)
        self.v.set_buffers(*[t.data_ptr() for t in (self.obs, self.fin, self.rew, self.term, self.trunc, self.info,
                                                   self.epr, self.epl)])
        self.host = {k_: getattr(self, k_).numpy() for k_ in ("term", "trunc", "info", "epr", "epl", "rew")}
        self._act_np = self.act.numpy()

    def reset(self):
        self.v.reset()
        return self.obs

    def step_async(self, actions):
        if actions is not self.act:
            self.act.copy_(actions)  # synchronous if cuda
        self.v.step_async(self._act_np)  # the C++ side copies the actions at once

    def step_wait(self):
        from .._native import lib
        lib().fm_step_wait(self.v.h)  # no per-env python work (FastVecEnv.step_wait builds info dicts)
        return StepOut(self.obs, self.rew, self.term, self.trunc, self.epr, self.epl, self.info[:, 0],
                       self.info[:, 1], host=self.host)

    def final_obs(self, idx):
        return self.fin[idx]

    @property
    def frames(self):
        return self.v.frames

    def close(self):
        self.v.close()


class CudaMarioBackend(EnvBackend):
    """fast_mario_env.cuda.CudaVecEnv behind the EnvBackend interface: CUDA tensors out, host=None.

    Same constructor as FastMarioBackend (n_threads / pin_cpus accepted and ignored). With the same seed and
    env_offset a run is step-for-step identical to the CPU backend given the same actions. Returned tensors
    are the env's own buffers, overwritten by the next step; consumers on the same CUDA stream are ordered
    automatically."""

    def __init__(self, level="1-1", n_envs=8, seed=0, sticky=0.25, noop_max=30, max_steps=2000,
                 n_threads=None, frame_stack=4, env_offset=0, pin_cpus=None, rom_path=None, device="cuda"):
        from ..cuda import CudaVecEnv
        self.env = CudaVecEnv(level, n_envs, seed=seed, sticky=sticky, noop_max=noop_max, max_steps=max_steps,
                              frame_stack=frame_stack, device=device, env_offset=env_offset, rom_path=rom_path)
        self.n_envs, self.device = n_envs, torch.device(device)
        e = self.env
        self._out = StepOut(e.obs, e.rew, e.term.view(torch.uint8), e.trunc.view(torch.uint8), e.ep_ret, e.ep_len,
                            e.info["x_pos"], e.info["flag_get"], host=None)

    def reset(self):
        return self.env.reset()

    def step_async(self, actions):
        self.env.step(actions)

    def step_wait(self):
        return self._out

    def final_obs(self, idx):
        return self.env.final_obs[idx]

    @property
    def frames(self):
        return self.env.frames

    def close(self):
        pass
