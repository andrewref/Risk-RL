"""
Multi-agent self-play training script for PyRisk using Stable-Baselines3 PPO.

Four independent environments are created with SubprocVecEnv.  
Each environment hosts a full 4-player game in which every player uses the
aggressive-strategy PPO agent (same shared network).  Checkpoints are written
every 5 000 timesteps so you can stop/restart training at will; each run
will resume from the last saved numeric checkpoint.
"""

import os
import glob
import re
import torch as th
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from risk_env_multi import FourPlayerRiskEnv   # your custom Gymnasium env

# -----------------------------------------------------------------------------
# Environment factory
# -----------------------------------------------------------------------------
N_ENVS = 4  # four parallel workers

def make_env() -> gym.Env:
    return FourPlayerRiskEnv()

# -----------------------------------------------------------------------------
# Main (must be guarded on Windows)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    th.set_num_threads(1)

    # vectorised env
    vec_env = VecMonitor(SubprocVecEnv([make_env for _ in range(N_ENVS)]))

    # checkpoint parameters
    SAVE_EVERY  = 50_000       # timesteps between saves
    NEW_STEPS   = 500_000      # how many new timesteps this run
    SAVE_FOLDER = "checkpoints_multi"
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # find existing numeric checkpoints
    ckpts = glob.glob(os.path.join(SAVE_FOLDER, "ppo_multi_*.zip"))
    pattern = re.compile(r"ppo_multi_(\d+)\.zip$")
    starts = []
    for p in ckpts:
        m = pattern.search(os.path.basename(p))
        if m:
            starts.append(int(m.group(1)))
    if starts:
        last_step = max(starts)
        print(f"Resuming from step {last_step}")
        model = PPO.load(
            os.path.join(SAVE_FOLDER, f"ppo_multi_{last_step}.zip"),
            env=vec_env,
            device="cuda",
        )
        start_step = last_step
    else:
        print("Starting fresh")
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            device="cuda",
            n_steps=4096,
            batch_size=2048,
            learning_rate=2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )
        start_step = 0

    # train for NEW_STEPS more timesteps
    end_step = start_step + NEW_STEPS
    for step in range(start_step, end_step, SAVE_EVERY):
        model.learn(total_timesteps=SAVE_EVERY, reset_num_timesteps=False)
        ckpt = step + SAVE_EVERY
        ckpt_path = os.path.join(SAVE_FOLDER, f"ppo_multi_{ckpt}.zip")
        model.save(ckpt_path)
        print(f"[checkpoint] saved → {ckpt_path}")

    # final friendly save
    final_path = os.path.join(SAVE_FOLDER, f"ppo_multi_final_{end_step}.zip")
    model.save(final_path)
    print(f"[final checkpoint] saved → {final_path}")
