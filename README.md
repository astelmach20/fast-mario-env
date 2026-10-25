# fast-mario-env

A bit-exact, batched, **10-15x faster** Super Mario Bros environment for reinforcement learning.

- **Same environment, faster.** A C++ re-implementation of the [nes-py](https://github.com/Kautenja/nes-py)
  NES emulator plus the [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) reward/info
  logic and the usual Atari-style wrappers (sticky actions, frame skip, max-pool, life loss, no-op starts,
  grayscale + 84x84 `INTER_AREA` resize, time limit, frame stack, auto-reset). RAM, screen, observations,
  rewards, dones and infos are identical to the nes-py stack **step for step**, so results stay comparable
  with work done on nes-py.
- **One process, N threads.** All envs live in your process and are stepped by a C++ thread pool with the
  GIL released; about 0.36 MB per env instead of tens of MB per worker process.
- **Optional CUDA backend.** `CudaVecEnv` runs tens of thousands of the same bit-exact envs on one NVIDIA GPU,
  compiled at runtime with NVRTC from torch's CUDA wheels (no nvcc needed).
- **Lean PPO trainer.** `fast_mario_env.ppo` implements the math of Stable-Baselines3's PPO (Nature CNN, same
  init and loss) with CUDA graphs and zero-copy env buffers: about 2x SB3's throughput with matching learning
  curves.
- **SB3 drop-ins.** `sb3_vec_env(...)` is an SB3 `VecEnv`; `DedupReplayBuffer` is a frame-deduplicating replay
  buffer for SB3 DQN that needs about 3.8x less memory than SB3's `optimize_memory_usage=True`.

No ROM is included: the game ROM is loaded at runtime from your gym-super-mario-bros installation or a path you
provide.

## Install

Linux x86-64, Python >= 3.10, a C++17 compiler (g++ or clang++). The native library is compiled during install.

```bash
pip install .                 # core: numpy only
pip install ".[sb3]"          # + gymnasium, stable-baselines3 (sb3_vec_env, DedupReplayBuffer)
pip install ".[cuda]"         # + torch (CudaVecEnv, FrameReplay); use a CUDA build of torch
pip install ".[train]"        # + torch, tensorboard (PPO trainer)
pip install ".[test]"         # + pytest and the reference stack (nes-py 8.2.1, gym-super-mario-bros 7.4.0)
```

Compiler selection: `$CXX` if set, else `c++` / `g++` / `clang++` from `PATH`. Without a system compiler,
`pip install ziglang` and either set `CXX="python -m ziglang c++"` or let the build pick it up automatically
when nothing else is found. `FAST_MARIO_ENV_MARCH` overrides the default `-march=x86-64-v3` (only used if the
build machine has AVX2; `none` disables it) and `FAST_MARIO_ENV_CXXFLAGS` adds flags.

### The ROM

The ROM is looked up in this order:

1. the `rom_path=` argument (`FastVecEnv(..., rom_path="/path/to/super-mario-bros.nes")`),
2. the `FAST_MARIO_ENV_ROM` environment variable,
3. the ROM files inside an installed `gym-super-mario-bros` package (found on disk, not imported).

A missing or unsupported ROM raises an error that lists these options. Only NROM (mapper 0) Super Mario Bros
images are supported; `rom_mode` selects gym-super-mario-bros' `vanilla` (default), `downsample`, `pixel` or
`rectangle` variant.

## Quickstart

```python
import numpy as np
from fast_mario_env import FastVecEnv

env = FastVecEnv("1-1", n_envs=32, seed=0, sticky=0.25, noop_max=30)
obs = env.reset()                                    # (32, 4, 84, 84) uint8
for _ in range(1000):
    obs, rew, terminated, truncated, infos = env.step(np.random.randint(0, 7, 32))
env.close()
```

Stable-Baselines3:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from fast_mario_env import sb3_vec_env

venv = VecFrameStack(VecMonitor(sb3_vec_env("1-1", 8, sticky=0.25, noop_max=30)), 4)  # (84, 84, 4) obs
# or, with fewer copies and the same values: stacked (4, 84, 84) obs straight from C++
venv = VecMonitor(sb3_vec_env("1-1", 8, stack_in_cpp=True, sticky=0.25, noop_max=30))
PPO("CnnPolicy", venv).learn(1_000_000)
```

GPU (needs torch with CUDA):

```python
import torch
from fast_mario_env.cuda import CudaVecEnv

env = CudaVecEnv("1-1", n_envs=32768, seed=0, sticky=0.25, noop_max=30)  # first use compiles (~1 min, cached)
obs = env.reset()                                                         # CUDA uint8 (N, 4, 84, 84)
obs, rew, term, trunc, info = env.step(torch.randint(0, 7, (32768,), device="cuda"))
```

PPO trainer (needs a CUDA GPU):

```bash
python -m fast_mario_env.ppo.train --preset sb3 --level 1-1 --total-steps 3e6 --out runs/1-1
python -m fast_mario_env.ppo.train --preset sb3 --bench 20        # throughput only
```

## API

`FastVecEnv(level="1-1", n_envs=8, seed=0, frame_stack=4, sticky=0.0, noop_max=0, max_steps=2000,
maxpool=False, skip=4, n_threads=None, copy_obs=True, rom_path=None, rom_mode="vanilla", monitor_path=None,
env_offset=0)`

| argument | meaning |
|---|---|
| `level` | `"w-s"` for all envs, or a list of N levels (one per env) |
| `seed`, `env_offset` | env i draws sticky actions and no-op counts from `np.random.default_rng(seed * 1000 + env_offset + i)` |
| `sticky` | per-NES-frame probability of repeating the previous action |
| `noop_max` | 0..`noop_max` random no-op agent steps after every reset |
| `max_steps` | time limit in agent steps (sets `truncated`) |
| `maxpool` | max over the last two skipped frames (in gray), as ALE does |
| `n_threads` | worker threads, default `min(n_envs, os.cpu_count())`; 0 or 1 steps inline |
| `copy_obs` | `False` returns a view of the internal obs buffer (valid until the next step) |
| `monitor_path` | write an SB3 `VecMonitor`-compatible `monitor.monitor.csv` |

- `reset() -> obs`: (N, k, 84, 84) uint8; stacks are zero-padded at episode start.
- `step(actions)`, or `step_async(actions)` then `step_wait()`, returns `obs, reward (float32), terminated,
  truncated, infos`. `step_async` returns immediately (the GIL is released), so a learner can work while the
  envs step.
- `infos[i]` has `x_pos, flag_get, life, score, time, world, stage`. For an env whose episode ended in this
  step it also has `episode = {r, l, t, x_pos, flag_get}`, `final_obs` (the last stacked obs before the
  automatic reset) and `TimeLimit.truncated`.
- An episode terminates on losing a life or reaching the flag. Actions are gym-super-mario-bros'
  `SIMPLE_MOVEMENT` (7 actions); rewards are gym-super-mario-bros' (x progress, clock, death penalty), summed
  over the skipped frames.
- Also `ram(i)` (2 KB RAM of env i), `frames` (total NES frames emulated), `bad_ops()`, `set_buffers(...)`
  (let the native side write into your own, e.g. pinned, buffers) and `close()`.

Other helpers: `sb3_vec_env(level, n_envs, seed, stack_in_cpp=False, **FastVecEnv kwargs)`, `RawNES` (a single
raw emulator: `frame(buttons, render)`, `ram`, `screen`, `backup()`, `restore()`), `resize_area`, `gray_lut`,
`palette_rgb`, `find_rom`, `decode_level`, `ALL_LEVELS`.

`fast_mario_env.cuda.CudaVecEnv(level, n_envs, seed, sticky, noop_max, max_steps, skip=4, device="cuda",
rom_path=None, env_offset=0)` has the same semantics with `frame_stack=4` and `maxpool=False`. Inputs and outputs
are CUDA tensors (`obs, rew, term, trunc, info`, where `info` is a dict of int32 tensors plus `ep_ret`, `ep_len`
and `final_obs`). Returned tensors are internal buffers that the next step overwrites, and nothing is copied to
the host. Given the same seed and actions it produces the same trajectories as `FastVecEnv`. See
[docs/cuda.md](docs/cuda.md).

`fast_mario_env.replay.FrameReplay(capacity_per_env, n_envs, k=4, device="cpu")` and
`fast_mario_env.replay.DedupReplayBuffer` (an SB3 `replay_buffer_class`): see
[docs/design.md](docs/design.md#replay-buffer).

## Fidelity

The reference is nes-py 8.2.1 + gym-super-mario-bros 7.4.0 (`SuperMarioBrosEnv` + `JoypadSpace(SIMPLE_MOVEMENT)`)
with the numpy/OpenCV wrapper stack in [tests/reference_env.py](tests/reference_env.py). "Bit-exact" means that,
given the same ROM, seed and actions, every step gives identical RAM, observations, rewards,
terminated/truncated, info values, episode statistics and `final_obs`. The tests check exactly that
(`pip install ".[test]"`, then `pytest`):

| test | what is compared |
|---|---|
| `test_fidelity_raw.py` | the NES core vs nes-py's own compiled emulator, with the same random controller input and random state backup/restore: full RAM after every frame, full RGB screen on every rendered frame (rendering on for a random 30% of frames, or for all frames) |
| `test_fidelity_resize.py` | gray LUT vs `cv2.cvtColor(RGB2GRAY)`; both resize paths vs `cv2.resize(INTER_AREA)` on random and NES-like images |
| `test_fidelity_env.py` | `FastVecEnv` vs the reference env stack, every step: obs, reward, terminated, truncated, all info keys, episode r/l, `final_obs`, `TimeLimit.truncated` and the RAM of every env. Configs: default, sticky + no-op (0.25 / 30), max-pool + sticky, short time limit; levels 1-1, 4-1 and a mixed-level batch |
| `test_cuda.py` | `CudaVecEnv` vs `FastVecEnv` in lockstep on 9 levels (RAM, obs, rewards, dones, infos, episode stats, `final_obs`), and the optimized vs the straightforward GPU kernel on all 32 levels (skipped without a GPU) |
| `test_replay.py`, `test_api.py` | replay sampling vs explicitly stored transitions; API, ROM handling, determinism across thread counts |

The default sizes keep CI to a few minutes. `FME_TEST_SCALE=10 pytest` runs 10x more frames and steps, and
`FME_TEST_POLICY=model.zip` drives `test_fidelity_env.py` with a trained SB3 agent instead of random actions.

Larger runs made during development, all with 0 mismatches and 0 unimplemented opcodes executed (raw data in
[benchmarks/results](benchmarks/results)):

- raw frames: 5k + 3x30k + 10k (every frame rendered) + 2x8k + 10k frames;
- env level: 1500 steps x 4 envs x 4 configs on 1-1 and 4-1, and the same driven by a trained PPO agent
  (73 episodes, 37 flag clears);
- CUDA vs CPU: all 32 levels side by side (512 envs x 1500 steps x 3 configs, about 42k episodes), and with a
  trained agent (16 envs x 1500 steps x 3 configs, 129 flag clears).

Not covered: the Lost Levels ROM. Random play also rarely reaches the later parts of levels; the flag, pipe and
time-limit paths were exercised with the trained agent on 1-1 only. Every level runs through the same mapper and
code paths, so the remaining risk is a rarely used opcode or PPU path. The core counts unimplemented opcodes
(`bad_ops()`), and that count stayed at 0 in all runs.

## Benchmarks

All numbers are from one machine: **AMD Ryzen 7 9800X3D (8C/16T), NVIDIA RTX 5070 Ti 16 GB**, Linux, torch 2.11
(CUDA 12.8). Unless noted: random actions, frame skip 4, sticky 0.25, no-op starts 30, level 1-1. Raw data and
measurement details are in [benchmarks/results](benchmarks/results) and the scripts in [benchmarks](benchmarks).

### CPU env-only throughput (agent steps/s)

Measured with `nice -n 19` while the machine was saturated by other training jobs (load average 6-20). The
baseline (nes-py, one worker process per env, shared-memory observations) and `FastVecEnv` (16 threads) were run
back to back under the same conditions.

| n_envs | nes-py, 1 process/env | FastVecEnv | speedup |
|---:|---:|---:|---:|
| 1 (single core) | 210 | 2,701 | **12.8x** |
| 8  | 789   | 7,857  | **10.0x** |
| 16 | 925   | 11,162 | **12.1x** |
| 32 | 942   | 14,245 | **15.1x** |
| 64 | 1,135 | 17,304 | **15.3x** |

On the same machine without other load (no other process above 30% CPU), `FastVecEnv` reached 3,544 steps/s on
one core, 18.9k with 8 envs, 24.5k with 16, 25.6k with 32, 27.6k with 64 and 29.1k with 128 envs. The baseline was
not re-measured under those conditions.

Memory: 0.36 MB per env (process PSS growth / N at N = 16-64; 4.5 KB of it is emulator state), against 27 MB PSS
(64.5 MB RSS) per nes-py worker process.

### End-to-end SB3 PPO (CnnPolicy, CUDA learner, same contended conditions; steps/s)

| n_envs | nes-py | FastVecEnv + VecFrameStack | FastVecEnv, `stack_in_cpp=True` |
|---:|---:|---:|---:|
| 8  | 683 | 2,852 (4.2x) | 3,560 (**5.2x**) |
| 32 | 923 | 4,989 (5.4x) | 6,413 (**6.9x**) |

The gain is smaller than env-only because once the emulator is fast, SB3's own per-step work (policy forward,
wrappers, rollout buffer) and the PPO update dominate. The lean trainer below addresses that.

### Lean PPO trainer (`fast_mario_env.ppo`, `--preset sb3`)

`--preset sb3` uses SB3's PPO math with these hyper-parameters: 8 envs, n_steps 512, batch 256, 4 epochs,
lr 2.5e-4, clip 0.1, ent 0.01, vf 0.5, gamma 0.9, lambda 0.95, max-grad-norm 0.5, fp32. Throughput on a quiet
machine:

| | steps/s |
|---|---:|
| SB3 PPO on FastVecEnv + VecFrameStack (8 envs) | 4,584 |
| lean trainer, eager (`--graph 0`) | 7,795 |
| **lean trainer, default (CUDA graphs)** | **9,240** (full 3M-step run: 9,633) |
| 4 / 6 / 8 concurrent runs under CUDA MPS (aggregate) | 19,563 / 21,800 / 22,986 |

A single 8-env run is bound by the lockstep env step: 58% of its wall time is spent waiting for the slowest env.
Several concurrent runs under MPS (see [docs/ppo.md](docs/ppo.md)) use the machine better.

Learning on 1-1 (flag rate over 100 stochastic evaluation episodes with sticky actions and no-op starts):

| run | 1M | 1.5M | 2M | 2.5M | 3M |
|---|---:|---:|---:|---:|---:|
| SB3 PPO, seed 0 | 0.01 | 0.22 | 0.69 | 0.75 | 0.70 |
| SB3 PPO, seed 1 | 0.00 | 0.15 | 0.38 | 0.72 | 0.77 |
| lean, seed 0 | 0.01 | 0.22 | 0.62 | 0.77 | 0.81 |
| lean, seed 1 | 0.01 | 0.05 | 0.37 | 0.66 | 0.75 |
| lean, seed 2 | 0.00 | 0.11 | 0.20 | 0.53 | 0.73 |

Mean flag rate at 3M steps: SB3 0.735 (2 seeds), lean 0.76 (3 seeds). The curves agree within seed noise.

### CUDA backend (`CudaVecEnv`, env-only, all 32 levels mixed)

**The GPU was 84-100% busy with another training job during every one of these measurements**, so they
understate an idle GPU.

| n_envs | steps/s | ms per step |
|---:|---:|---:|
| 256 | 2.1k | 119 |
| 1,024 | 8.4k | 122 |
| 4,096 | 32.0k | 128 |
| 16,384 | 83.3k | 197 |
| 32,768 | 142k | 231 |
| 65,536 | 145k | 452 |

At 32,768 envs on 1-1 only: 154k steps/s (random actions) and 180k (the same action in all envs). Best observed,
with the GPU about 25% loaded by other work: 233k steps/s (random actions) and 302k (same action) at 32,768 envs.

**The GPU backend is latency-bound below about 16k envs.** One env step is about 120 ms of serial work for one GPU
thread (4 frames, roughly 17k emulated 6502 instructions plus PPU work and a render), so until the GPU is full,
throughput grows only with the number of envs. It overtakes the 16-thread CPU backend (17-29k steps/s) only at
a few thousand envs, and gives a large gain only at 16k+ envs with a learner that consumes that many envs per
step; for typical 8-128 env setups use `FastVecEnv`. Memory is 85 KB per env (65,536 envs = 5.4 GB). Details in
[docs/cuda.md](docs/cuda.md).

### Replay buffer (bytes per transition, 4x84x84 uint8 stacks)

| buffer | bytes / transition | 1M transitions |
|---|---:|---:|
| SB3 `ReplayBuffer` | 56,468 | 56.5 GB |
| SB3 `optimize_memory_usage=True` | 28,244 | 28.2 GB |
| `FrameReplay` / `DedupReplayBuffer` | **7,421** | **7.4 GB** |

Sampling a batch of 256 with 3-step returns takes 0.94 ms on CUDA and 3.2 ms on CPU.

## Design

Profiling nes-py's env showed about 85% of each step in the emulator, and 80% of that in the PPU (stepped dot by
dot, three times per CPU cycle, through `std::function` bus maps). fast-mario-env keeps the emulated behaviour and
changes how it is computed:

- **Catch-up PPU**: the PPU only advances when the CPU touches a PPU register, an NMI may be due, or the frame
  ends. Whole scanline segments advance with closed-form scroll updates, and sprite-0 hit is resolved exactly
  inside its window.
- **Render only observed frames** (the last frame of each skip, the last two with max-pool), with span-based
  rendering and tile and sprite-line caches.
- **CPU**: a 256-way templated switch, a flat NROM bus, and an exact skip of SMB's idle loop (about 55% of
  executed instructions).
- **Exact ports of the Python/OpenCV stages**: grayscale via a palette LUT, OpenCV's float `INTER_AREA` (same
  operation order, compiled with `-ffp-contract=off`), numpy's PCG64 for sticky actions and no-ops, the SMB
  reward/info logic, frame stack and auto-reset.
- **One process, N threads, no IPC**. The console state is a plain 4.5 KB struct (snapshot = memcpy), which is
  also what made the CUDA port practical.

More, including the CUDA kernel design and the remaining performance headroom, in [docs/design.md](docs/design.md)
and [docs/cuda.md](docs/cuda.md).

## Limitations

- Super Mario Bros 1 only (NROM / mapper 0), standard levels 1-1 to 8-4; no Lost Levels and no random-stage env.
- The action set is `SIMPLE_MOVEMENT` (7 actions); observations are 84x84 grayscale (no RGB observations).
- Linux x86-64 is the tested platform. The code has a portable (non-SSE) path, but other platforms are untested.
  The CUDA backend needs an NVIDIA GPU, torch with CUDA, and Linux (`libcuda.so.1`).
- The CUDA backend supports `frame_stack=4`, `maxpool=False` and `rom_mode="vanilla"` only. It needs about a
  minute to compile on first use, plus 9-25 s to boot the levels each time an env is constructed.
- The tests' reference stack needs nes-py 8.2.1 with numpy < 2 (nes-py 8.2.1 fails with numpy 2).
- Fidelity is defined against nes-py 8.2.1, including its quirks, not against real NES hardware.

## Credits and license

MIT licensed, copyright (c) 2026 Andrew Stelmach; see [LICENSE](LICENSE).

- The emulator core is a re-implementation of the [nes-py](https://github.com/Kautenja/nes-py) emulator by
  Christian Kauten (MIT), which is itself based on [SimpleNES](https://github.com/amhndu/SimpleNES) by Amish
  Naidu.
- Reward, info and level semantics follow [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
  by Christian Kauten.
- The resize and grayscale ports reproduce [OpenCV](https://opencv.org)'s `INTER_AREA` and `RGB2GRAY`, and the
  sticky-action / no-op RNG reproduces numpy's PCG64.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project is not affiliated with or endorsed by Nintendo.
Super Mario Bros is a trademark of Nintendo; you need your own legally obtained ROM.

If you use this in research, please cite:

```bibtex
@software{stelmach2026fastmarioenv,
  author = {Stelmach, Andrew},
  title  = {fast-mario-env: a bit-exact, batched Super Mario Bros environment for reinforcement learning},
  year   = {2026},
  url    = {https://github.com/astelmach20/fast-mario-env}
}
```
