import sys
import os
import glob
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from risk_env_multi import FourPlayerRiskEnv
from stable_baselines3.common.vec_env import DummyVecEnv

class FourPlayerAggressiveEnv(FourPlayerRiskEnv):
    """
    Aggressive strategy PPO-compatible environment:
    - Rewards frequent attacks
    - Greatly rewards conquest
    - Penalizes passivity
    """
    def _update_action_space(self):
        import gymnasium as gym
        return gym.spaces.Discrete(max(1, len(self.action_list)))

    def step(self, act_idx):
        reward = -0.1  # discourage stalling

        if act_idx < len(self.action_list):
            atk, tgt = self.action_list[act_idx]
            prev_owner = self.world.owner(tgt)
            success = self.world.attack(atk, tgt)

            if success:
                reward += 1.0
                if prev_owner != self.world.owner(tgt):
                    reward += 3.0
            else:
                reward -= 0.5
        else:
            reward -= 0.5

        done = self.world.game_over()
        self.world.next_player()
        self.action_list = self._legal_attacks()
        self.action_space = self._update_action_space()

        if not self.action_list and not done:
            done = True
        if done:
            reward += 50.0

        return self._get_obs(), reward, done, False, {}

# Incremental PPO training with automatic resume
if __name__ == "__main__":
    CHECKPOINT_FOLDER = "checkpoints/aggressive"
    FINAL_MODEL_PATH = os.path.join(CHECKPOINT_FOLDER, "ppo_aggressive_final.zip")
    TIMESTEP_FILE = os.path.join(CHECKPOINT_FOLDER, "timesteps.txt")
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    env = DummyVecEnv([lambda: FourPlayerAggressiveEnv()])

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
