# Design notes

## What is reproduced

`FastVecEnv` reproduces, bit for bit, this stack (implemented in Python in `tests/reference_env.py`):

```
nes-py SuperMarioBrosEnv(target=(w, s)) + JoypadSpace(SIMPLE_MOVEMENT)
-> sticky actions (per NES frame, numpy PCG64 Generator per env, seeded seed*1000+i)
-> frame skip k (rewards summed, early stop on done; optional gray max-pool of the last 2 frames)
-> life loss / flag_get end the episode
-> random no-op starts (0..noop_max agent steps of action 0 after reset)
-> cv2 RGB2GRAY + cv2.resize(84x84, INTER_AREA)
-> time limit (truncation)
-> k-frame stack (zero-padded at episode start) + auto-reset with final_obs / episode infos
```

nes-py quirks are kept on purpose: an instruction-atomic CPU with skip cycles, nes-py's PPU sprite-0 and
scrolling model, the bit-reversed status-flag byte layout produced by its bitfield, NT3 mirroring of `$3000`, and
so on. Note also that in nes-py every step returns the same screen buffer, so a "max-pool" over two frames taken
from it naively is a no-op; `maxpool=True` here is a real gray max over the last two frames (as ALE does) and the
default `maxpool=False` reproduces the common naive wrapper.

## Where the time went in the reference

Per agent step (about 5 ms): the emulator is about 85%; inside it the PPU is about 80% (dot by dot, three times
per CPU cycle, through `std::function` bus maps). Then cv2 preprocessing (about 7%), Python reward/info (about 3%)
and inter-process communication for the vectorized env.

## How the C++ backend is faster

1. **Catch-up PPU.** The PPU only advances when the CPU touches PPU registers, at the NMI point, or at frame end.
   Scanlines are advanced in bulk with closed-form scroll increments; the sprite-0 hit is resolved exactly, pixel
   by pixel, but only inside its window.
2. **Rendering only when an observation needs it**: the last frame of each skip (the last two with max-pool).
   Early termination restores a snapshot and re-runs that frame with rendering on. Rendering is span-based with
   tile and sprite-line caches.
3. **CPU**: a 256-way templated switch, a flat NROM bus, and an exact skip of SMB's `JMP *` idle loop (in 3-cycle
   multiples, stopping at the NMI or frame end), which removes about 55% of executed instructions.
4. **Exact C++ ports of the remaining Python/cv2 stages**: `RGB2GRAY` applied to a palette LUT; `INTER_AREA` ported
   with OpenCV's float operation order (vectorized, with an SSE2 transpose; built with `-ffp-contract=off` so no
   FMA contraction changes rounding); numpy PCG64 `random()` / `integers()`; SMB reward and info; frame stack and
   auto-reset.
5. **One process, N threads, no IPC.** The console is one POD `NesState` (2 KB RAM, 2 KB VRAM, OAM, palette,
   registers; about 4.5 KB), so snapshot/restore is a memcpy. A whole env is about 349 KB (mostly 16 snapshots for
   sticky / early termination and three 240x256 gray buffers).

An rdtsc profile (under contention) gives about 0.2M ticks per non-rendered frame, of which the PPU is only about
20% (the rest is CPU interpretation), and about 0.7M ticks per rendered frame, about 70% of it pixel output.

## Remaining headroom (not implemented on the CPU)

- Fast-forwarding SMB's sprite-0 poll loop (`$8150`, about 170 iterations per frame) the same exact way as the idle
  loop (the CUDA kernel does this).
- A threaded-code or decoded-instruction cache for the interpreter (about 40 cycles per instruction now).
- Rendering only rows that survive the 84x84 area resize (all rows contribute today, so this needs a fused
  render + resize).

With the emulator about 15x faster, an SB3 training loop becomes learner-bound (policy forward, wrapper copies,
rollout buffer, update); `stack_in_cpp=True` recovers 25-30% of that, and the lean PPO trainer
(`docs/ppo.md`) most of the rest.

## Replay buffer

`fast_mario_env.replay.FrameReplay` stores each 84x84 frame once in a ring per env, plus per-slot metadata (age since
episode start, action, reward, done/terminal flags), instead of two k-frame stacks per transition.

- Stacks are rebuilt at sample time by gathering the k most recent slots and zero-masking anything before the
  episode start, which reproduces the zero-padded reset stacks exactly.
- Terminal frames are stored explicitly (one extra slot per episode), so time-limit truncations bootstrap
  correctly. SB3's `optimize_memory_usage` overwrites the terminal next_obs with the next reset obs (SB3 issue
  #934), which is why SB3 forbids combining it with timeout handling.
- Supports n-step returns and discounts (Rainbow / MR.Q style) and returns env/slot indices for prioritized
  replay updates.
- Storage can live on the GPU (`device="cuda"`); 1M transitions take 7.4 GB.
- `DQN(..., replay_buffer_class=fast_mario_env.replay.DedupReplayBuffer)` works with SB3 DQN (1-step,
  timeout-aware; observations must be channel-first, which is what SB3 stores after `VecTransposeImage` or with
  `sb3_vec_env(stack_in_cpp=True)`).
