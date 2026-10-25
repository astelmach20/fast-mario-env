"""Env-only throughput: FastVecEnv (C++ threads) vs the reference nes-py stack (one process per env).

    python benchmarks/bench_env.py --ns 1,8,16,32,64 --secs 10            # both
    python benchmarks/bench_env.py --ns 64 --skip-baseline                # FastVecEnv only

Random actions, frame skip 4, sticky 0.25, no-op starts 30, level 1-1 (the settings of the README
table). The baseline needs nes-py + gym-super-mario-bros + opencv (the [test] extra); it runs
tests/reference_env.RefMarioEnv in N worker processes (the README numbers used an equivalent
shared-memory multiprocess wrapper). Prints one JSON line per measurement.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import fast_mario_env as fme

TESTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")


def _worker(conn, level, seed, kw):
    sys.path.insert(0, TESTS)
    from reference_env import RefMarioEnv
    env = RefMarioEnv(level, seed=seed, **kw)
    env.reset()
    while True:
        a = conn.recv()
        if a is None:
            break
        obs, r, te, tr, _ = env.step(a)
        if te or tr:
            obs = env.reset()
        conn.send((obs, r, te, tr))


class ProcVecEnv:
    def __init__(self, level, n, seed, kw):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, level, seed * 1000 + i, kw), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(int(a))
        return [c.recv() for c in self.conns]

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(5)


def timed(step, n, secs):
    rng = np.random.default_rng(0)
    for _ in range(20):
        step(rng.integers(0, 7, n))
    t, k = time.time(), 0
    while time.time() - t < secs:
        step(rng.integers(0, 7, n))
        k += n
    return k / (time.time() - t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="1,8,16,32,64")
    ap.add_argument("--secs", type=float, default=10)
    ap.add_argument("--threads", type=int, default=None, help="FastVecEnv threads (default min(N, cpus))")
    ap.add_argument("--sticky", type=float, default=0.25)
    ap.add_argument("--noop", type=int, default=30)
    ap.add_argument("--level", default="1-1")
    ap.add_argument("--skip-baseline", action="store_true")
    a = ap.parse_args()
    kw = dict(sticky=a.sticky, noop_max=a.noop)
    for n in map(int, a.ns.split(",")):
        res = {}
        if not a.skip_baseline:
            ref = ProcVecEnv(a.level, n, 0, kw)
            res["reference"] = timed(ref.step, n, a.secs)
            ref.close()
        threads = a.threads or min(n, os.cpu_count() or 1)
        v = fme.FastVecEnv(a.level, n, seed=0, n_threads=threads, **kw)
        v.reset()
        res["fast"] = timed(v.step, n, a.secs)
        bpe = v.bytes_per_env()
        v.close()
        out = dict(n_envs=n, threads=threads, fast_steps_per_s=round(res["fast"]), bytes_per_env=bpe,
                   load1=round(os.getloadavg()[0], 1), **kw)
        if "reference" in res:
            out.update(reference_steps_per_s=round(res["reference"]), speedup=round(res["fast"] / res["reference"], 2))
        print(json.dumps(out), flush=True)
