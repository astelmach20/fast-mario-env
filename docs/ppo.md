# Lean PPO trainer

`python -m fast_mario_env.ppo.train` (`src/fast_mario_env/ppo/`) is a single-file, CleanRL-style PPO for this env.
Needs torch with CUDA (`pip install ".[train]"`). Raw measurements: `benchmarks/results/ppo_trainer.jsonl`;
learning curves: `benchmarks/results/ppo_learning_summary.json`. Hardware: Ryzen 7 9800X3D (8C/16T), RTX 5070 Ti,
torch 2.11 cu128. "Quiet" below means no other CUDA context and no other process above 30% CPU.

## `--preset sb3` (default): SB3 PPO math

- 8 envs, n_steps 512, batch 256, 4 epochs
- lr 2.5e-4, clip 0.1, ent 0.01, vf 0.5, gamma 0.9, lambda 0.95, max-grad-norm 0.5, Adam eps 1e-5
- Nature CNN with SB3's orthogonal init (shared features, separate linear policy / value heads)
- per-minibatch advantage normalization, unclipped value loss, timeout bootstrap (`r += gamma * V(final_obs)`)
- env: sticky 0.25, no-op 30, time limit 2000; fp32 (TF32 allowed), synchronous (no policy lag)

The speedups come only from systems work:

- CUDA graphs for the policy step, the whole PPO minibatch step (gather, forward, backward, grad clip, fused
  capturable Adam) and GAE;
- the C++ env writes directly into pinned torch tensors (no numpy or Python per-env work per step);
- rollout storage on the GPU, no Python wrappers in the loop.

```bash
python -m fast_mario_env.ppo.train --preset sb3 --level 1-1 --seed 0 --total-steps 3e6 --out runs/sb3eq_s0
python -m fast_mario_env.ppo.train --preset sb3 --bench 20          # throughput only, no outputs
```

Outputs in `--out`: TensorBoard scalars (`mario/x_pos_mean100`, `mario/flag_rate100`, `mario/best_x_pos`,
`mario/episodes`, PPO stats, `eval/flag_rate`), `monitor.monitor.csv`, `config.json`, `progress.jsonl` (evaluation
every 500k steps: one stochastic episode in each of 100 parallel envs with the training perturbations),
`result.json` (with timers), and a `model.pt` checkpoint at every evaluation.

Other options: `--groups N` (double-buffered env groups that keep the RNG streams of a single env batch),
`--env-backend cuda` (GPU emulator; the trainer then keeps dones and episode stats on the device and copies them
once per rollout), `--device-env-test 1` (exercises the device path with the CPU env), `--bf16 1`,
`--async-update 1`, `--target 0.9` (stop when the evaluation flag rate reaches it).

## Where SB3 spends its time (quiet, 8 envs)

SB3 PPO on `sb3_vec_env` + VecMonitor + VecFrameStack runs at 4,584 steps/s. Per env step:

| part | us/step | share |
|---|---:|---:|
| C++ emulator | 59.4 | 27% |
| Python env/wrapper overhead | 28.5 | 13% |
| policy forward (eager) | 53.9 | 25% |
| rollout buffer (numpy) | 16.1 | 7% |
| PPO update | 48.1 | 22% |
| host-to-device and other | 12.0 | 6% |

With 32 envs SB3 reaches 6,697 steps/s: more envs amortize the forward pass, but wrappers, buffer and update stay
at about the same cost per step.

## Single-run speed (quiet, 20-30 s runs)

| variant | steps/s | notes |
|---|---:|---|
| SB3 (reference) | 4,584 | |
| lean, `--graph 0` (eager) | 7,795 | |
| **lean default (graphs, fp32)** | **9,240** (full 3M run: **9,633**) | **2.1x SB3** |
| `--groups 2` | 9,106 | does not help: the lockstep env step dominates |
| bf16 + groups 2 (with an experimental spinning worker pool) | 9,783 | not default: bf16 not verified to give the same learning |

Timers of the full 3M-step run (311.7 s of training wall time): waiting for the env 180.8 s (58%), waiting for the
GPU 38 s, update 66.8 s (21%). A single 8-env run is therefore bound by the lockstep env step: a batch step of 8
envs takes about 420-750 us while one env takes about 280 us, because the batch waits for the slowest env plus
thread wake-up latency. Spinning pinned workers, CPU pinning, SMT-aware placement and double-buffered groups did
not help; C-state exit latency (up to 350 us with the powersave governor) is the likely remaining cause.

## Aggregate throughput: K concurrent runs (levels cycled; quiet; 30 s each)

| K | no MPS | with user-level MPS |
|---:|---:|---:|
| 1 | 9,435 | 9,301 |
| 2 | 12,697 | 13,812 |
| 3 | 14,597 | |
| 4 | 15,979 | **19,563** |
| 6 | 15,065 | **21,800** |
| 8 | 17,917 | **22,986** |

Without MPS, separate processes time-slice the GPU by context, so each run's small batch-8 graph launches
serialize behind other runs' update graphs; MPS lets them overlap. 6-8 concurrent runs under MPS give about
22-23k aggregate steps/s: 2.4x a single run, about 5x SB3, and about 77% of the CPU emulator bound below. Each
run's config and math are unchanged.

```bash
source benchmarks/mps.sh start      # user-level MPS daemon, no root; exports CUDA_MPS_PIPE_DIRECTORY
for lv in 1-1 1-2 4-1 6-1; do
  python -m fast_mario_env.ppo.train --preset sb3 --level $lv --out runs/$lv > runs/$lv.log 2>&1 &
done
wait; source benchmarks/mps.sh stop
```

## Ceilings (quiet)

`FastVecEnv` env-only (steps/s): 3,544 (1 env, 1 core), 18.9k (8 envs, 8 threads) / 19.9k (16 threads), 24.5k (16),
25.6k (32), 27.6k (64), 29.1k (128). The CPU emulator bound on this machine is about 29-30k steps/s.

Learner (forward + backward + fused Adam, Nature CNN, samples/s): fp32 eager 200k at batch 256; fp32 + graph 212k
(b256), about 229k (b4096); bf16 + channels_last + graph 257k (b256), 265k (b1024), 272-277k (b4096). Inference
(graph) takes about 0.1 ms per batch-8 step and runs at about 600k samples/s at batch 256. The learner-bound PPO
rate at 4 epochs is then about 1 / (4/250k + 1/600k), i.e. 55-62k env steps/s, which is the ceiling a GPU emulator
plugged into the trainer would hit.

## Learning matches SB3 (1-1; flag rate over 100 stochastic evaluation episodes)

| run | 0.5M | 1M | 1.5M | 2M | 2.5M | 3M |
|---|---:|---:|---:|---:|---:|---:|
| SB3 seed 0 (`benchmarks/sb3_baseline.py`) | 0.00 | 0.01 | 0.22 | 0.69 | 0.75 | 0.70 |
| SB3 seed 1 | 0.00 | 0.00 | 0.15 | 0.38 | 0.72 | 0.77 |
| **lean seed 0** | 0.00 | 0.01 | 0.22 | 0.62 | 0.77 | **0.81** |
| **lean seed 1** | 0.01 | 0.01 | 0.05 | 0.37 | 0.66 | **0.75** |
| lean seed 2 | 0.00 | 0.00 | 0.11 | 0.20 | 0.53 | 0.73 |

Mean flag rate at 3M: SB3 0.735 (2 seeds), lean 0.78 (seeds 0/1) and 0.76 (3 seeds). Steps to a 50% flag rate are
2.0-2.5M for both; the curves agree within seed noise. (The wall-clock times of these learning runs are not
meaningful because they shared the machine with other jobs; use the quiet numbers above.)

## A bug worth knowing about

CUDA graphs captured on the shared default capture stream share the cuBLAS workspace (cached per handle and
stream). When an inference graph and an update graph replayed concurrently (async mode), they corrupted each
other and produced NaN log-probs / values at random steps. Fix: every graph is captured on its own stream
(`torch.cuda.graph(g, stream=s)`). As safety nets the log-ratio is clamped to +-20 and a minibatch with a
non-finite gradient is skipped (counted as `train/nonfinite_skips`, which has stayed 0 since the fix).

## `--preset exp_fast` (EXPERIMENTAL; changes the training math)

128 envs in 2 groups, n_steps 64, batch 512, bf16, asynchronous update with one rollout of policy lag.

| run | 2M | 3M | 3.5M | 4M | 4.5M | 5M | steps/s | wall to 70% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp_fast seed 0 | 0.03 | 0.28 | 0.57 | 0.90 | 0.80 | 0.88 | 24.4k | 163 s |
| exp_fast seed 1 | 0.06 | 0.40 | 0.61 | 0.79 | 0.71 | 0.83 | 22.2k | 173 s |
| batch 256, seed 0 | 0.42 | 0.71 | 0.83 | 0.93 | 0.86 | 0.85 | 20.6k | 145 s |
| batch 256, seed 1 | 0.29 | 0.80 | 0.60 | 0.68 | 0.87 | 0.94 | 20.9k | 152 s |
| 256 envs, seed 0 | 0.12 | 0.36 | 0.59 | 0.67 | 0.89 | 0.73 | 20.9k | 217 s |
| no async, seed 0 | 0.17 | 0.71 | 0.78 | 0.84 | **0.00** | 0.00 | | |

These reach a 70% flag rate in about 2.5 minutes of wall clock (vs about 4.3 minutes for one `--preset sb3` run)
but need more env steps (3-4.5M) and are less stable: the no-async run collapsed (entropy collapse after a
value-loss spike at about 4.3M steps). Do not use them without per-level validation.
