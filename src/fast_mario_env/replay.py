# SPDX-License-Identifier: GPL-3.0-or-later
"""Frame-deduplicating replay buffer (needs torch) for frame-stacked pixel envs (DQN / Rainbow / MR.Q style).

Stores each 84x84 frame ONCE per env (ring of C slots), plus per-slot metadata, instead of
two full k-frame stacks per transition.  Stacks are rebuilt at sample time by gathering the
k most recent slots and zeroing those that precede the episode start (exactly reproducing the
zero-padded reset stack of FastVecEnv / SB3 VecFrameStack).

Terminal frames are stored explicitly (one extra slot per episode), so truncation is handled
correctly -- unlike SB3's optimize_memory_usage, where the terminal next_obs is overwritten by the
next episode's reset obs (SB3 GH #934) and timeout handling has to be disabled.

Storage is torch tensors on any device: device="cuda" keeps the whole buffer on the GPU, so
sampling is a GPU gather and no host->device copy of observations happens per batch.

    rb = FrameReplay(capacity_per_env, n_envs, k=4, device="cuda")
    rb.add(obs, next_obs, actions, rewards, dones, truncated)   # obs/next_obs: (N, k, 84, 84) uint8
    s = rb.sample(256, n_step=3, gamma=0.99)   # s.obs, s.action, s.reward (n-step), s.next_obs, s.discount

SB3:  DQN(..., replay_buffer_class=fast_mario_env.replay.DedupReplayBuffer)  (obs must be channel-first (k,84,84),
which is what SB3 stores after its automatic VecTransposeImage.)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch as th


class Batch(NamedTuple):
    obs: th.Tensor        # (B, k, H, W) uint8
    action: th.Tensor     # (B,) int64
    reward: th.Tensor     # (B,) float32, n-step discounted sum
    next_obs: th.Tensor   # (B, k, H, W) uint8 (state after m <= n steps)
    discount: th.Tensor   # (B,) float32 = gamma^m * (1 - terminal)
    terminal: th.Tensor   # (B,) float32 (true termination, not truncation)
    steps: th.Tensor      # (B,) int64 m
    env: th.Tensor        # (B,) env index
    pos: th.Tensor        # (B,) absolute slot index (for PER priority updates)


class FrameReplay:
    def __init__(self, capacity_per_env: int, n_envs: int, k: int = 4, hw=(84, 84), device="cpu", seed=None):
        self.C, self.E, self.k, self.hw = int(capacity_per_env), int(n_envs), int(k), tuple(hw)
        self.dev = th.device(device)
        E, C, d = self.E, self.C, self.dev
        self.frames = th.zeros((E, C, *self.hw), dtype=th.uint8, device=d)
        self.age = th.zeros((E, C), dtype=th.uint8, device=d)      # frames since episode start (sat. 255)
        self.act = th.zeros((E, C), dtype=th.int16, device=d)
        self.rew = th.zeros((E, C), dtype=th.float32, device=d)
        self.flags = th.zeros((E, C), dtype=th.uint8, device=d)    # 1=valid start, 2=done, 4=terminal
        self.ptr = np.zeros(E, np.int64)                           # absolute #slots written per env
        self.new_ep = np.ones(E, bool)
        self.g = th.Generator(device=d)
        self.g.manual_seed(int(np.random.SeedSequence(seed).generate_state(1)[0]))
        self._eidx = th.arange(E, device=d)
        self.n_transitions = 0

    # ------------------------------------------------------------------ add
    def _t(self, x, dtype=None):
        if isinstance(x, th.Tensor):
            return x.to(self.dev, dtype=dtype, non_blocking=True)
        return th.as_tensor(np.ascontiguousarray(x), device=self.dev, dtype=dtype)

    def _write(self, envs: np.ndarray, frame: th.Tensor, age: th.Tensor):
        s = th.as_tensor(self.ptr[envs] % self.C, device=self.dev)
        e = th.as_tensor(envs, device=self.dev)
        self.frames[e, s] = frame
        self.age[e, s] = age
        self.flags[e, s] = 0
        self.ptr[envs] += 1

    def add(self, obs, next_obs, action, reward, done, truncated=None):
        """obs/next_obs: (N, k, H, W) stacks (next_obs = terminal stack for done envs).
        done = terminated | truncated; truncated marks time-limit ends (bootstrapped)."""
        done = np.asarray(done, bool).reshape(-1)
        trunc = np.zeros_like(done) if truncated is None else np.asarray(truncated, bool).reshape(-1)
        allv = np.arange(self.E)
        ne = np.flatnonzero(self.new_ep)
        if len(ne):  # first frame of a new episode (stack is zeros + this frame)
            o = self._t(obs)[th.as_tensor(ne, device=self.dev), -1]
            self._write(ne, o, th.zeros(len(ne), dtype=th.uint8, device=self.dev))
        s = th.as_tensor((self.ptr - 1) % self.C, device=self.dev)
        e = self._eidx
        self.act[e, s] = self._t(action).reshape(-1).to(th.int16)
        self.rew[e, s] = self._t(reward, th.float32).reshape(-1)
        fl = 1 | (2 * done) | (4 * (done & ~trunc))
        self.flags[e, s] = self._t(fl.astype(np.uint8))
        age = th.clamp(self.age[e, s].to(th.int16) + 1, max=255).to(th.uint8)
        self._write(allv, self._t(next_obs)[:, -1], age)
        self.new_ep = done.copy()
        self.n_transitions += self.E

    # --------------------------------------------------------------- sample
    def size(self):
        return int(min(self.n_transitions, self.E * self.C))

    def _stack(self, e, p):
        k = self.k
        off = th.arange(k - 1, -1, -1, device=self.dev)            # k-1 .. 0 frames back
        idx = (p[:, None] - off[None]) % self.C
        fr = self.frames[e[:, None], idx]                          # (B, k, H, W)
        keep = off[None] <= self.age[e, p % self.C].long()[:, None]
        return fr * keep[:, :, None, None].to(th.uint8)

    def sample(self, batch: int, n_step: int = 1, gamma: float = 0.99) -> Batch:
        B, C, k, d = batch, self.C, self.k, self.dev
        ptr = th.as_tensor(self.ptr, device=d)
        lo = th.clamp(ptr - C + k - 1, min=0)                      # lookback stays inside live ring
        hi = ptr - 1 - n_step                                      # p+n-1 has its next frame written
        ok_env = th.nonzero(hi >= lo).flatten()
        if len(ok_env) == 0:
            raise ValueError("not enough data in buffer")
        e = th.empty(B, dtype=th.long, device=d)
        p = th.empty(B, dtype=th.long, device=d)
        todo = th.arange(B, device=d)
        for _ in range(64):                                        # rejection: skip terminal-frame slots
            n = len(todo)
            ee = ok_env[th.randint(len(ok_env), (n,), device=d, generator=self.g)]
            span = (hi[ee] - lo[ee] + 1).double()
            pp = lo[ee] + (th.rand(n, device=d, generator=self.g, dtype=th.float64) * span).long()
            good = (self.flags[ee, pp % C] & 1).bool()
            e[todo[good]], p[todo[good]] = ee[good], pp[good]
            todo = todo[~good]
            if len(todo) == 0:
                break
        else:
            raise RuntimeError("sampling failed")
        R = th.zeros(B, device=d)
        alive = th.ones(B, device=d)
        term = th.zeros(B, device=d)
        m = th.zeros(B, dtype=th.long, device=d)
        for i in range(n_step):
            s = (p + i) % C
            f = self.flags[e, s]
            R += alive * (gamma ** i) * self.rew[e, s]
            m += alive.long()
            term = th.maximum(term, alive * ((f & 4) > 0).float())
            alive = alive * ((f & 2) == 0).float()
        disc = (gamma ** m.double()).float() * (1 - term)
        return Batch(self._stack(e, p), self.act[e, p % C].long(), R, self._stack(e, p + m), disc, term, m, e, p)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.frames, self.age, self.act, self.rew, self.flags))


def _sb3_buffer_class():
    from stable_baselines3.common.buffers import BaseBuffer
    from stable_baselines3.common.type_aliases import ReplayBufferSamples

    class DedupReplayBuffer(BaseBuffer):
        """Drop-in replay_buffer_class for SB3 DQN/QR-DQN (1-step, timeout-aware)."""

        def __init__(self, buffer_size, observation_space, action_space, device="auto", n_envs=1,
                     optimize_memory_usage=False, handle_timeout_termination=True, storage_device=None):
            super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
            k, h, w = observation_space.shape
            per_env = max(buffer_size // n_envs, 1)
            self.rb = FrameReplay(per_env + per_env // 20 + k + 2, n_envs, k, (h, w),
                                  device=storage_device or self.device)
            self.cap_transitions = per_env * n_envs
            self.handle_timeout_termination = handle_timeout_termination

        def add(self, obs, next_obs, action, reward, done, infos):
            tr = np.array([bool(i.get("TimeLimit.truncated", False)) for i in infos]) \
                if self.handle_timeout_termination else None
            self.rb.add(obs, next_obs, action, reward, done, tr)
            self.pos = self.rb.n_transitions // self.n_envs
            self.full = self.rb.n_transitions >= self.cap_transitions

        def size(self):
            return min(self.rb.n_transitions, self.cap_transitions) // self.n_envs

        def reset(self):
            super().reset()

        def sample(self, batch_size, env=None):
            b = self.rb.sample(batch_size, 1, 1.0)
            dv = self.device
            return ReplayBufferSamples(b.obs.to(dv), b.action.to(dv)[:, None], b.next_obs.to(dv),
                                       b.terminal.to(dv)[:, None], b.reward.to(dv)[:, None])

        def _get_samples(self, batch_inds, env=None):
            raise NotImplementedError

    return DedupReplayBuffer


_CLS = {}


def __getattr__(name):
    if name == "DedupReplayBuffer":
        if name not in _CLS:
            _CLS[name] = _sb3_buffer_class()
        return _CLS[name]
    raise AttributeError(name)
