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

# Training example (OPTIONAL use in train script)
if __name__ == "__main__":
    env = DummyVecEnv([lambda: FourPlayerAggressiveEnv()])
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_aggressive")
