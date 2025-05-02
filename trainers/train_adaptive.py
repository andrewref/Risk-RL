#!/usr/bin/env python3
import sys
import os
import glob
import re
import torch as th

# ─── Make sure our project root is on PYTHONPATH ──────────────────────────
# (so `import strategies.reward_adaptive` will work)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from strategies.reward_adaptive import FourPlayerAdaptiveEnv

# ─── Configuration ───────────────────────────────────────────────────────
STRATEGY_NAME    = "adaptive"
CHECKPOINT_FOLDER= os.path.join("checkpoints", STRATEGY_NAME)
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_FOLDER, f"ppo_{STRATEGY_NAME}_final.zip")
TIMESTEP_FILE    = os.path.join(CHECKPOINT_FOLDER, "timesteps.txt")
INCREMENT        = 50_000

if __name__ == "__main__":
    # Create folder if missing
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    # Build a DummyVecEnv with our Adaptive env
    env = DummyVecEnv([lambda: FourPlayerAdaptiveEnv()])

    print(f"Using {'cuda' if th.cuda.is_available() else 'cpu'} device")

    # ─── Load or initialize model ──────────────────────────
    if os.path.exists(FINAL_MODEL_PATH):
        print("Resuming training from final checkpoint...")
        model = PPO.load(
            FINAL_MODEL_PATH,
            env=env,
            device="cuda" if th.cuda.is_available() else "cpu",
        )
        # Read how many timesteps we've done so far
        if os.path.exists(TIMESTEP_FILE):
            with open(TIMESTEP_FILE, "r") as f:
                total_steps = int(f.read())
        else:
            total_steps = 0
    else:
        print("Starting new adaptive model from scratch...")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            device="cuda" if th.cuda.is_available() else "cpu",
            n_steps=2048,
            batch_size=1024,
            learning_rate=2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )
        total_steps = 0

    target_steps = total_steps + INCREMENT

    # ─── Training loop ────────────────────────────────────────
    while total_steps < target_steps:
        model.learn(total_timesteps=INCREMENT, reset_num_timesteps=False)
        total_steps += INCREMENT

        # Save updated model and timestep count
        model.save(FINAL_MODEL_PATH)
        with open(TIMESTEP_FILE, "w") as f:
            f.write(str(total_steps))

        print(f"[Checkpoint] Saved at {total_steps} total steps to {FINAL_MODEL_PATH}")

    print(f"[Training complete] Final adaptive model saved → {FINAL_MODEL_PATH}")
