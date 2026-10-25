# CUDA backend

`fast_mario_env.cuda.CudaVecEnv` is a CUDA port of the C++ core (`csrc/nes_core.hpp`) and env layer
(`csrc/fastmario.cpp`) with one CUDA thread per env. It is step-for-step bit-exact with `FastVecEnv`: RAM, obs,
rewards, terminated/truncated and infos, on all 32 levels, with sticky actions, no-op resets and auto-reset.
Observations, rewards, dones and infos stay in CUDA tensors; nothing is copied to the host per step.

## Build and dependencies

- **Compiler.** NVRTC comes from torch's CUDA wheels (`nvidia-cuda-nvrtc-cu12`), and the driver API (`libcuda.so.1`)
  is used through ctypes (`cujit.py`). No nvcc or host compiler is needed. Set `FAST_MARIO_ENV_NVRTC` to a
  `libnvrtc.so` path to override the lookup.
- **When.** Kernels are compiled on first construction (about a minute) and the cubin is cached in
  `$FAST_MARIO_ENV_CACHE/cuda` (default `~/.cache/fast_mario_env/cuda`), keyed by source, options, GPU
  architecture and NVRTC version.
- **Options.** `--fmad=false --device-int128 -std=c++17 -DSKIP_K=<skip>`. FMA contraction is off so the
  INTER_AREA float math matches the CPU exactly; 128-bit integers are needed for PCG64.
- **Construction** boots each level on the GPU and builds a per-level reset cache (about 9 s for one level, 25 s
  for all 32).

## API

```python
from fast_mario_env.cuda import CudaVecEnv
env = CudaVecEnv(level="1-1" or ["1-1", "4-2", ...], n_envs=32768, seed=0, sticky=0.25, noop_max=30,
                 max_steps=2000, device="cuda", env_offset=0)
obs = env.reset()                                  # uint8 (N, 4, 84, 84) CUDA
obs, rew, term, trunc, info = env.step(actions)    # actions: int64 (N,) CUDA; outputs are CUDA tensors
# rew float32 (N,), term/trunc bool (N,), info = dict(x_pos=int32 (N,), flag_get=..., life, score, time,
# world, stage, coins, y_pos, status, ep_ret, ep_len, final_obs)
# env.final_obs / env.ep_ret / env.ep_len are valid where term | trunc
```

- Env i uses RNG stream `seed*1000 + env_offset + i`, as `FastVecEnv` does, so given the same actions a GPU run is
  identical to a CPU run.
- Everything is queued on torch's current stream with no host synchronization. Returned tensors are the env's own
  buffers and are overwritten by the next step (`clone()` what you need to keep).
- `ram()` returns the (N, 2048) RAM of every env (a copy; for tests), `bad_ops()` the number of unimplemented
  opcodes executed, `bytes_per_env()` the device memory per env.
- For the PPO trainer: `python -m fast_mario_env.ppo.train --env-backend cuda --n-envs 4096 --n-steps 64 --batch 4096 --epochs 2`
  (see `fast_mario_env/ppo/backends.py`, `CudaMarioBackend`). A demo run with 4096 envs, n_steps 64, batch 4096,
  2 epochs, on a GPU about 80% loaded by another job, reached 18.7k steps/s end to end (1.31M steps in 70 s).

## Performance

Env-only, random actions generated on the GPU, frame skip 4, sticky 0.25, no-op 30, RTX 5070 Ti
(`benchmarks/bench_cuda.py`; raw data `benchmarks/results/cuda_throughput.jsonl`). **The GPU was 84-100% busy with
another training job during every measurement**; contexts are time-sliced, so even a small concurrent job cuts
this kernel's share. A rerun with only light concurrent jobs gave the same numbers (98k at 16k envs, 143k at 32k,
146k at 65k).

| n_envs | all 32 levels mixed | 1-1 random | 1-1 same action in all envs | ms/step |
|---:|---:|---:|---:|---:|
| 256 | 2.1k | | | 119 |
| 1,024 | 8.4k | | | 122 |
| 4,096 | 32.0k | | | 128 |
| 16,384 | 83.3k | | | 197 |
| 32,768 | 142k | 154k | 180k | 231 |
| 65,536 | 145k | 161k | 183k | 452 |

- Best observed: at 32,768 envs with lower external load (about 25%), 233k steps/s with random actions and 302k
  with the same action in all envs.
- For comparison, the 16-thread CPU backend does 17-29k steps/s (contended vs quiet), so the GPU gives about
  6-9x env-only under heavy contention, and up to about 13x with a less loaded GPU.
- **Latency floor.** One env step is about 120 ms of serial work for one thread (4 frames, roughly 17k 6502
  instructions plus PPU catch-up and a render). Below about 16k envs the GPU is latency-bound, so throughput
  scales linearly with n_envs; a learner needs 16k+ envs per step for the GPU emulator to pay off.
- **Memory**: 85.3 KB per env (65,536 envs = 5.4 GB): obs, final_obs and the new-frame buffer are 63 KB, the
  deferred-render log about 12.6 KB, emulator state plus step snapshot about 9 KB; the level cache is shared.

Per-kernel time at 32,768 envs (GPU about 97% busy), random actions: `k_step2` (emulation) 75-130 ms,
`k_render2` 45 ms, `k_stack` 4-7 ms, `k_reset` about 0 ms per step.

Why it does not go faster: each env runs its own 6502 program path (warp divergence; random actions cost about 2x
vs converged actions even with the forced-reconvergence design below); `k_step2` uses 128 registers, limiting
occupancy; rendering is 30-40% of step time. Ideas not implemented: grouping envs per warp by level and phase,
warp-cooperative PPU catch-up, caching the render log in shared memory, CUDA graphs for the 4 launches.

## Kernel design

**v1 (`kernels/nes_cuda.cu`)** is a direct device port of the C++ core and env layer. Each env's state is
interleaved per 32-bit word (`S[w*N + e]`) for coalescing. It is kept as the GPU reference (`kernel="v1"`) and is
also used for `k_boot` and `k_cache`, which build per-level reset snapshots after the exact title/no-op sequence.

**v2 (`kernels/nes_cuda2.cu`)**, the default, all exact:

- **Uniform fast path.** A 256-entry micro-op decode table (read, ALU, write, flags, control) runs every official
  non-I/O opcode branch-free; I/O, BRK/RTI/JMP-indirect and unofficial opcodes take the slow switch.
- **Lazy sprite evaluation.** Only the sprite-0 flag is computed per line; the full list is materialized on
  OAM/`$2000`/`$2003`/`$2004`/`$4014` writes, on render and at state store.
- **Poll skip.** SMB's sprite-0 poll loop (`LDA $2002; AND #$40; BEQ/BNE`) is fast-forwarded exactly, bounded by
  the next candidate sprite-0 hit, pre-render clear, NMI and frame end. It accounted for 80% of slow instructions.
- **Flattened state machine.** `k_step2` is a single loop over an attempt / frame-start / run / frame-end / done
  state machine with a warp vote per iteration to force reconvergence (3.7x faster than the nested form).
- **Deferred render.** On the observed frame, emulation only logs line segments (y, x0, x1, v, fine_x, PPU flags,
  sprite-eval record, VRAM/OAM versions) into a per-env log; VRAM, palette and OAM are copied on the first write
  of the frame. A write landing between already-logged segments, or a log overflow, replays the whole step from
  a snapshot with inline rendering (this path is part of what the fidelity tests exercise).
- **`k_render2`**: persistent 256-thread blocks with CHR in shared memory; each thread draws an 8-dot span of one
  row; 8 rows per phase, then INTER_AREA 240x256 -> 84x84 with the exact CPU float operation order.
- **`k_reset`** applies the cached level reset after rendering (deferred auto-reset); **`k_stack`** updates the
  frame stack, final_obs and episode return/length.

## Fidelity record

All with 0 mismatches and 0 unimplemented opcodes (`benchmarks/results/cuda_fidelity.jsonl`): all 32 levels side
by side, 512 envs x 1500 steps, configs default / sticky 0.25 + no-op 30 / short time limit (12,467 + 12,581 +
17,470 episodes); a trained PPO agent on 1-1, 16 envs x 1500 steps x the same 3 configs (83 + 83 + 400 episodes,
68 + 61 flag clears); v2 vs v1 on all 32 levels after every optimization. `tests/test_cuda.py` runs a smaller
version of the CPU-vs-GPU and v2-vs-v1 checks.
