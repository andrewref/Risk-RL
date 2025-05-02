from stable_baselines3 import PPO
from risk_env_multi import FourPlayerRiskEnv
import gymnasium as gym
import numpy as np
import random

class FourPlayerRandomEnv(FourPlayerRiskEnv):
    """
    Random strategy PPO-compatible environment:
    - Picks a random valid attack when available
    - Rewards attacks with some variability
    """
    def step(self, act_idx):
        reward = -0.1

        if self.action_list:
            # Use random strategy regardless of the given act_idx
            atk_idx = random.randint(0, len(self.action_list) - 1)
            atk, tgt = self.action_list[atk_idx]
            prev_owner = self.world.owner(tgt)
            success = self.world.attack(atk, tgt)

            if success:
                reward += 0.5
                if prev_owner != self.world.owner(tgt):
                    reward += 1.5
            else:
                reward -= 0.3
        else:
            reward -= 0.2

        done = self.world.game_over()
        self.world.next_player()
        self.action_list = self._legal_attacks()
        self.action_space = self._update_action_space()

        if not self.action_list and not done:
            done = True
        if done:
            reward += 50.0

        return self._get_obs(), reward, done, False, {}

    def _update_action_space(self):
        # Ensures action space is always valid and dynamic
        return gym.spaces.Discrete(max(1, len(self.action_list)))
