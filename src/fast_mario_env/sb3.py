"""Stable-Baselines3 VecEnv adapter over FastVecEnv (needs `pip install fast-mario-env[sb3]`)."""
from __future__ import annotations

import numpy as np

from .env import N_ACTIONS, OBS_HW, FastVecEnv


def sb3_vec_env(level="1-1", n_envs=8, seed=0, stack_in_cpp=False, **kw):
    """SB3 ``VecEnv`` over :class:`FastVecEnv`.

    stack_in_cpp=False (default): single frames, channels-last (84, 84, 1) obs, the same layout as
        gym-super-mario-bros + an SB3 ``WarpFrame``; wrap with ``VecMonitor`` + ``VecFrameStack(venv, 4)``
        as usual.
    stack_in_cpp=True: (k, 84, 84) channels-first stacked obs straight from C++. Skip VecFrameStack /
        VecTransposeImage: the values are the ones SB3 would build, with fewer copies.

    Remaining keyword arguments go to FastVecEnv (sticky, noop_max, max_steps, maxpool, ...).
    """
    from gymnasium import spaces
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    k = kw.pop("frame_stack", 4) if stack_in_cpp else 1

    class FastSB3VecEnv(VecEnv):
        def __init__(self):
            self.v = FastVecEnv(level, n_envs, seed=seed, frame_stack=k, **kw)
            shape = (k, OBS_HW, OBS_HW) if stack_in_cpp else (OBS_HW, OBS_HW, 1)
            super().__init__(n_envs, spaces.Box(0, 255, shape, np.uint8), spaces.Discrete(N_ACTIONS))

        def _fmt(self, o):
            return o if stack_in_cpp else o[:, 0, :, :, None]

        def reset(self):
            return self._fmt(self.v.reset())

        def step_async(self, actions):
            self.v.step_async(actions)

        def step_wait(self):
            obs, r, te, tr, infos = self.v.step_wait()
            for i in infos:
                if "final_obs" in i:
                    fo = i.pop("final_obs")
                    i["terminal_observation"] = fo if stack_in_cpp else fo[0][:, :, None]
                    i.pop("episode", None)  # VecMonitor computes its own
            return self._fmt(obs), r, te | tr, infos

        def close(self):
            self.v.close()

        def get_attr(self, name, indices=None):
            return [None] * len(self._idx(indices))

        def set_attr(self, name, value, indices=None):
            pass

        def env_method(self, name, *a, indices=None, **k_):
            return [None] * len(self._idx(indices))

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * len(self._idx(indices))

        def seed(self, seed=None):
            return [None] * self.num_envs

        def _idx(self, indices):
            return range(self.num_envs) if indices is None else ([indices] if isinstance(indices, int) else indices)

    return FastSB3VecEnv()
