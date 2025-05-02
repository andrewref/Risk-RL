import sys
import os
import numpy as np
from stable_baselines3 import PPO
from risk_env_multi import FourPlayerRiskEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces

# Ensure project root is on path so checkpoints can be loaded
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FourPlayerAdaptiveEnv(FourPlayerRiskEnv):
    """
    Adaptive PPO-compatible environment:
    Dynamically delegates actions to one of four pretrained PPO agents
    (aggressive, defensive, balanced, random) based on game state every 7 rounds.
    """
    def __init__(self):
        super().__init__()
        self.strategy = "random"
        self.round_counter = 0
        # Load pretrained strategy agents
        self.strategy_models = {
            "aggressive": PPO.load(
                os.path.join("checkpoints", "aggressive", "ppo_aggressive_final.zip"), env=self
            ),
            "defensive": PPO.load(
                os.path.join("checkpoints", "defensive", "ppo_defensive_final.zip"), env=self
            ),
            "balanced": PPO.load(
                os.path.join("checkpoints", "balanced", "ppo_balanced_final.zip"), env=self
            ),
            "random": PPO.load(
                os.path.join("checkpoints", "random", "ppo_random_final.zip"), env=self
            ),
        }

    def _update_action_space(self):
        return spaces.Discrete(max(1, len(self.action_list)))

    def step(self, _):
        # Delegate action selection to the chosen strategy model
        # Reward shaping is handled by underlying PPO models
        self.round_counter += 1
        if self.round_counter % 7 == 0:
            self.strategy = self._select_strategy()

        obs = self._get_obs()
        # Prepare observation for model.predict
        obs_input = {"troops": obs["troops"].astype(np.float32), "pid": obs["pid"]}
        # Predict action from selected strategy agent
        model = self.strategy_models[self.strategy]
        action, _ = model.predict(obs_input, deterministic=True)
        # Ensure action is within legal range
        action = int(action)
        reward = -0.1

        if action < len(self.action_list):
            atk, tgt = self.action_list[action]
            prev_owner = self.world.owner(tgt)
            success = self.world.attack(atk, tgt)
            # Basic reward for conquest
            if success and prev_owner != self.world.owner(tgt):
                reward += 1.0
        else:
            # No-op penalty
            reward -= 0.2

        done = self.world.game_over()
        self.world.next_player()
        self.action_list = self._legal_attacks()
        self.action_space = self._update_action_space()

        # End-of-episode reward
        if done:
            reward += 50.0

        return self._get_obs(), reward, done, False, {}

    def _select_strategy(self):
        # Decide which pretrained agent to use based on board control
        my_terrs = self.world.my_territories()
        total = len(self.world.territories)
        owned = len(my_terrs)
        borders = sum(
            1 for t in my_terrs if any(self.world.is_enemy(n) for n in self.world.get_neighbors(t))
        )
        ratio = owned / total
        if ratio > 0.35:
            return "aggressive"
        elif ratio < 0.25 and borders >= 5:
            return "defensive"
        elif any(t in ["Ukraine", "Northern Europe", "Middle East", "China"] for t in my_terrs):
            return "balanced"
        else:
            return "random"

# Optional standalone test
if __name__ == "__main__":
    env = DummyVecEnv([lambda: FourPlayerAdaptiveEnv()])
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_adaptive_final.zip")



