import sys
import os
import glob
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from strategies.reward_balanced import FourPlayerBalancedEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Incremental PPO training with automatic resume
if __name__ == "__main__":
    CHECKPOINT_FOLDER = "checkpoints/balanced"
    FINAL_MODEL_PATH = os.path.join(CHECKPOINT_FOLDER, "ppo_balanced_final.zip")
    TIMESTEP_FILE = os.path.join(CHECKPOINT_FOLDER, "timesteps.txt")
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    env = DummyVecEnv([lambda: FourPlayerBalancedEnv()])

    # Resume from latest checkpoint or start new
    if os.path.exists(FINAL_MODEL_PATH):
        print(f"Resuming training from final checkpoint...")
        model = PPO.load(FINAL_MODEL_PATH, env=env)
        if os.path.exists(TIMESTEP_FILE):
            with open(TIMESTEP_FILE, "r") as f:
                total_steps = int(f.read())
        else:
            total_steps = 0
    else:
        print("Starting new model from scratch...")
        model = PPO("MultiInputPolicy", env, verbose=1)
        total_steps = 0

    # Training loop with continuous update
    INCREMENT = 50000
    TARGET_TOTAL = total_steps + INCREMENT

    while total_steps < TARGET_TOTAL:
        model.learn(total_timesteps=INCREMENT, reset_num_timesteps=False)
        total_steps += INCREMENT
        model.save(FINAL_MODEL_PATH)
        with open(TIMESTEP_FILE, "w") as f:
            f.write(str(total_steps))
        print(f"[Checkpoint] Model saved at {total_steps} total steps")

    print("[Training complete] Final model saved.")
