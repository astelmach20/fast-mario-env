# Raw benchmark and fidelity data

Raw JSON lines behind the tables in the top-level README. All measured on one machine: AMD Ryzen 7
9800X3D (8 cores / 16 threads), NVIDIA RTX 5070 Ti (16 GB), Linux, torch 2.11 (CUDA 12.8). The
machine was shared with other jobs during many of these runs; `load1` (1-minute load average) and,
for GPU runs, `smi_before` / `smi_after` (GPU utilisation from nvidia-smi) are recorded per entry.

| file | contents |
|---|---|
| `cpu_env.jsonl` | env-only throughput and memory, `FastVecEnv` (`kind: fastemu`) vs a multiprocess nes-py stack (`kind: shm_nespy`); end-to-end SB3 PPO (`bench: ppo_e2e`); replay-buffer size and speed |
| `ppo_trainer.jsonl` | lean PPO trainer throughput grids, learner ceilings, concurrent runs with and without MPS |
| `ppo_learning_summary.json` | flag-rate curves (100-episode evals every 0.5M steps) of the lean trainer and SB3 runs |
| `cuda_throughput.jsonl` | `CudaVecEnv` env-only throughput vs number of envs |
| `cuda_fidelity.jsonl` | CUDA vs CPU lockstep fidelity runs (all 32 levels; trained-policy runs) |

Some field names predate the package rename: `fastemu` / `fastmario` = this package's C++ backend,
`fastenv` / `shm_nespy` = the nes-py reference stack, `spin` / `spinpool` = an experimental worker
pool that was not faster and is not included here.
