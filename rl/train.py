# rl/ppo.py
"""
PPO training script for PyRisk with hardcoded opponents.
Trains a PPO agent using `env.py` where the agent plays as seat 0 ("ALPHA")
against 3 hardcoded bots (Aggressive, Balanced, Defensive).

Make sure to install:
  pip install stable-baselines3[extra] gym numpy
"""
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from rl.env import RiskEnv

# ─── Training Configuration ─────────────────────────────────────────── #
TOTAL_TIMESTEPS = 1_000_000
MODEL_DIR = "checkpoints"
MODEL_NAME = "ppo_risk"
SAVE_FREQ = 50_000  # Save every n steps

# ─── Ensure save directory exists ───────────────────────────────────── #
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Create the environment ─────────────────────────────────────────── #
env = DummyVecEnv([lambda: RiskEnv(seed=42)])

# ─── Initialize PPO model ───────────────────────────────────────────── #
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    ent_coef=0.01,
    learning_rate=3e-4,
    clip_range=0.2,
    tensorboard_log="runs/ppo_risk/"
)

# ─── Callback to save model every SAVE_FREQ steps ───────────────────── #
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODEL_DIR,
    name_prefix=MODEL_NAME
)

# ─── Begin training ─────────────────────────────────────────────────── #
print("\n[INFO] Starting PPO training...")
start_time = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
end_time = time.time()
print(f"\n[INFO] Training completed in {end_time - start_time:.2f} seconds.")

# ─── Save final model ───────────────────────────────────────────────── #
model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_final"))
print("[INFO] Final model saved.")
