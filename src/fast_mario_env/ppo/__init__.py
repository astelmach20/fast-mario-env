"""Lean PPO trainer (needs ``pip install fast-mario-env[train]``): the math of SB3's PPO with the Nature CNN,
made fast with CUDA graphs, GPU rollout storage and zero-copy env buffers.

    python -m fast_mario_env.ppo.train --preset sb3 --level 1-1 --total-steps 3e6 --out runs/1-1
"""
