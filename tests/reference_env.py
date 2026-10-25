"""Reference implementation used by the fidelity tests: nes-py + gym-super-mario-bros + numpy/cv2 wrappers.

This is the behaviour fast_mario_env reproduces bit for bit. It is deliberately simple, in-process
and slow (about 200 agent steps/s per env).

RefMarioEnv: SuperMarioBrosEnv(target=(w, s)) + JoypadSpace(SIMPLE_MOVEMENT), then
    sticky actions (per NES frame, from a numpy PCG64 Generator)
    -> frame skip (sum of rewards, stop early on done; optional gray max-pool of the last 2 frames)
    -> life loss and flag_get end the episode
    -> random no-op starts (0..noop_max agent steps of action 0 after reset)
    -> grayscale (cv2 RGB2GRAY) + cv2.resize(84x84, INTER_AREA)
    -> time limit (truncation after max_steps agent steps)
RefVecEnv: N RefMarioEnv with a k-frame stack (zero-padded at episode start), auto-reset, and
    'final_obs' / 'episode' / 'TimeLimit.truncated' infos, matching FastVecEnv's interface.
"""
import time
import warnings

import numpy as np

OBS_HW = 84
INFO_KEYS = ("x_pos", "flag_get", "life", "score", "time", "world", "stage")


class RefMarioEnv:
    def __init__(self, level="1-1", skip=4, max_steps=2000, sticky=0.0, noop_max=0, maxpool=False, seed=0):
        warnings.filterwarnings("ignore")
        import cv2
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
        from nes_py.wrappers import JoypadSpace

        cv2.setNumThreads(1)
        self.cv2 = cv2
        world, stage = (int(x) for x in level.split("-"))
        self.env = JoypadSpace(SuperMarioBrosEnv(target=(world, stage)), SIMPLE_MOVEMENT)
        self.nes = self.env.unwrapped
        self.skip, self.max_steps = skip, max_steps
        self.sticky, self.noop_max, self.maxpool = float(sticky), int(noop_max), bool(maxpool)
        self.rng = np.random.default_rng(seed)
        self.prev_action, self.lives = 0, None
        self.t, self.ep_ret, self.ep_len = 0, 0.0, 0

    @property
    def ram(self):
        return self.nes.ram

    def _gray(self, rgb):
        return self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)

    def _nes_frame(self, action):
        if self.sticky and self.rng.random() < self.sticky:
            action = self.prev_action
        self.prev_action = action
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
        else:
            obs, reward, term, info = out
            trunc = False
        return obs, float(reward), bool(term), bool(trunc), info

    def _agent_step(self, action):
        total, gray, gray_prev = 0.0, None, None
        for _ in range(self.skip):
            obs, r, term, trunc, info = self._nes_frame(action)
            total += r
            if self.maxpool:
                gray_prev, gray = gray, self._gray(obs)
            if term or trunc:
                break
        if self.maxpool and gray_prev is not None:
            gray = np.maximum(gray, gray_prev)
        life = info.get("life")
        if self.lives is not None and life is not None and life < self.lives:
            term = True
        self.lives = life
        if info.get("flag_get"):
            term = True
        return gray, total, term, trunc, info

    def _observe(self, gray):
        if gray is None:  # nes-py returns the live screen buffer; the last frame is what is observed
            gray = self._gray(self.nes.screen)
        return self.cv2.resize(gray, (OBS_HW, OBS_HW), interpolation=self.cv2.INTER_AREA)

    def _hard_reset(self):
        self.env.reset()
        self.prev_action, self.lives = 0, None

    def reset(self):
        self._hard_reset()
        if self.noop_max:
            for _ in range(int(self.rng.integers(0, self.noop_max + 1))):
                _, _, term, trunc, _ = self._agent_step(0)
                if term or trunc:
                    self._hard_reset()
        self.t, self.ep_ret, self.ep_len = 0, 0.0, 0
        return self._observe(None)

    def step(self, action):
        gray, r, term, trunc, info = self._agent_step(int(action))
        self.t += 1
        if self.t >= self.max_steps:
            trunc = True
        self.ep_ret += r
        self.ep_len += 1
        return self._observe(gray), r, term, trunc, info


class RefVecEnv:
    def __init__(self, level="1-1", n_envs=2, seed=0, frame_stack=4, **kw):
        levels = [level] * n_envs if isinstance(level, str) else list(level)
        self.envs = [RefMarioEnv(levels[i], seed=seed * 1000 + i, **kw) for i in range(n_envs)]
        self.n, self.k = n_envs, frame_stack
        self.stack = np.zeros((n_envs, frame_stack, OBS_HW, OBS_HW), np.uint8)
        self.t0 = time.time()

    def reset(self):
        self.stack[:] = 0
        for i, e in enumerate(self.envs):
            self.stack[i, -1] = e.reset()
        return self.stack.copy()

    def step(self, actions):
        rew = np.zeros(self.n, np.float32)
        term = np.zeros(self.n, bool)
        trunc = np.zeros(self.n, bool)
        infos = []
        self.stack[:, :-1] = self.stack[:, 1:]
        for i, e in enumerate(self.envs):
            obs, r, te, tr, info = e.step(int(actions[i]))
            rew[i], term[i], trunc[i] = r, te, tr
            info = {k: info[k] for k in INFO_KEYS if k in info}
            self.stack[i, -1] = obs
            if te or tr:
                info["final_obs"] = self.stack[i].copy()
                info["TimeLimit.truncated"] = bool(tr and not te)
                info["episode"] = dict(r=round(float(e.ep_ret), 6), l=int(e.ep_len), t=round(time.time() - self.t0, 6),
                                       x_pos=info.get("x_pos", 0), flag_get=bool(info.get("flag_get", False)))
                self.stack[i] = 0
                self.stack[i, -1] = e.reset()
            infos.append(info)
        return self.stack.copy(), rew, term, trunc, infos

    def ram(self, i):
        return self.envs[i].ram
