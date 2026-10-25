"""SB3 PPO reference run, for comparing learning curves with the lean trainer (fast_mario_env.ppo.train).

    python benchmarks/sb3_baseline.py --seed 0 --out runs/sb3_ref_s0

Same hyper-parameters as `--preset sb3`, the same env (FastVecEnv via sb3_vec_env + VecMonitor +
VecFrameStack(4), sticky 0.25, no-op 30, time limit 2000) and the same evaluation (one stochastic
episode in each of 100 parallel envs every 500k steps). Needs the [sb3] and [train] extras.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor

import fast_mario_env as fme


def evaluate(model, level, episodes, sticky, noop_max, seed):
    v = fme.FastVecEnv(level, n_envs=episodes, seed=seed, sticky=sticky, noop_max=noop_max)
    obs, res = v.reset(), {}
    while len(res) < episodes:
        a, _ = model.predict(obs, deterministic=False)
        obs, _, te, tr, infos = v.step(a)
        for i, inf in enumerate(infos):
            if (te[i] or tr[i]) and i not in res:
                res[i] = (float(inf["episode"]["x_pos"]), bool(inf["episode"]["flag_get"]))
    v.close()
    xs = [r[0] for r in res.values()]
    return dict(flag_rate=float(np.mean([r[1] for r in res.values()])), x_mean=float(np.mean(xs)), x_max=float(np.max(xs)))


ap = argparse.ArgumentParser()
ap.add_argument("--level", default="1-1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=float, default=3e6)
ap.add_argument("--eval-every", type=int, default=500_000)
ap.add_argument("--eval-episodes", type=int, default=100)
ap.add_argument("--n-envs", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(2)
os.makedirs(a.out, exist_ok=True)
venv = fme.sb3_vec_env(a.level, a.n_envs, seed=a.seed, sticky=0.25, noop_max=30)
venv = VecFrameStack(VecMonitor(venv, os.path.join(a.out, "monitor"), info_keywords=("x_pos", "flag_get")), 4)
model = PPO("CnnPolicy", venv, learning_rate=2.5e-4, n_steps=512, batch_size=256, n_epochs=4, gamma=0.9,
            gae_lambda=0.95, clip_range=0.1, ent_coef=0.01, vf_coef=0.5, seed=a.seed, verbose=0,
            device="cuda" if torch.cuda.is_available() else "cpu", tensorboard_log=a.out)
t0, ev_s = time.time(), 0.0
while model.num_timesteps < a.steps:
    model.learn(a.eval_every, reset_num_timesteps=False, progress_bar=False, tb_log_name="PPO")
    te = time.time()
    ev = evaluate(model, a.level, a.eval_episodes, 0.25, 30, a.seed + model.num_timesteps)
    ev_s += time.time() - te
    rec = dict(steps=int(model.num_timesteps), wall_s=round(time.time() - t0, 1),
               train_wall_s=round(time.time() - t0 - ev_s, 1), **ev)
    with open(os.path.join(a.out, "progress.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("EVAL", json.dumps(rec), flush=True)
model.save(os.path.join(a.out, "model.zip"))
