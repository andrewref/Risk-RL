# reward_defensive.py

import gymnasium as gym
import numpy as np
from pyrisk.world import World

class FourPlayerDefensiveEnv(gym.Env):
    """Defensive PPO agent: avoids battles unless opponent has only 1 troop."""

    def __init__(self):
        super().__init__()
        self.world = World()
        self.action_list = []
        n_t = len(self.world.territories)

        self.observation_space = gym.spaces.Dict({
            "troops": gym.spaces.Box(low=0, high=100, shape=(2, n_t), dtype=np.int32),
            "pid": gym.spaces.Discrete(4),
        })

        self.action_space = gym.spaces.Discrete(1)

    def _legal_attacks(self):
        acts = []
        for atk in self.world.my_territories():
            for nbr in self.world.get_neighbors(atk):
                if (self.world.is_enemy(nbr)
                        and self.world.troops(nbr) == 1
                        and self.world.troops(atk) > 2):
                    acts.append((atk, nbr))
        return acts

    def _get_obs(self):
        pid = self.world.current_player
        n_t = len(self.world.territories)
        my = np.zeros(n_t, dtype=np.int32)
        en = np.zeros(n_t, dtype=np.int32)
        for idx, terr in enumerate(self.world.territories):
            troops = self.world.troops(terr)
            if self.world.owner(terr) == pid:
                my[idx] = troops
            else:
                en[idx] = troops
        return {"troops": np.stack([my, en]), "pid": pid}

    def _update_action_space(self):
        return gym.spaces.Discrete(max(1, len(self.action_list)))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.action_list = self._legal_attacks()
        self.action_space = self._update_action_space()
        return self._get_obs(), {}

    def step(self, act_idx):
        reward = -0.1
        if act_idx < len(self.action_list):
            atk, tgt = self.action_list[act_idx]
            prev_owner = self.world.owner(tgt)
            success = self.world.attack(atk, tgt)
            if success and prev_owner != self.world.owner(tgt):
                reward = 1.0

        done = self.world.game_over()
        self.world.next_player()
        self.action_list = self._legal_attacks()
        self.action_space = self._update_action_space()

        if not self.action_list and not done:
            done = True
        if done:
            reward += 50.0

        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(self.world.get_map())
        print(self.world.last_event())
