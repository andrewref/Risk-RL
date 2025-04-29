# Gymnasium wrapper so SB3 can train four bots in parallel
import gymnasium as gym
import numpy as np
from pyrisk.world import World


class FourPlayerRiskEnv(gym.Env):
    """A very small action-space, self-play environment for Risk (4 bots)."""

    # ----------------------------------------------------------
    def __init__(self):
        super().__init__()

        # build a fresh PyRisk world (the wrapper hides all globals)
        self.world = World()                 # no args!

        # ----- SB3 spaces --------------------------------------------------
        self.action_list = []
        n_t = len(self.world.territories)

        # observation: 2×N array (my troops / enemy troops)  + current pid
        self.observation_space = gym.spaces.Dict({
            "troops": gym.spaces.Box(low=0, high=100, shape=(2, n_t),
                                     dtype=np.int32),
            "pid":    gym.spaces.Discrete(4),
        })

        # action: choose an index in self.action_list  (filled each turn)
        self.action_space = gym.spaces.Discrete(1)

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------
    def _legal_attacks(self):
        acts = []
        for atk in self.world.my_territories():
            for nbr in self.world.get_neighbors(atk):
                if (self.world.is_enemy(nbr)
                        and self.world.troops(atk) > self.world.troops(nbr) + 1):
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

    # ----------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.action_list = self._legal_attacks()
        self.action_space = gym.spaces.Discrete(max(1, len(self.action_list)))
        return self._get_obs(), {}

    def step(self, act_idx):
        # 1) perform chosen attack (or noop if index overflow)
        reward = -0.1
        if act_idx < len(self.action_list):
            atk, tgt = self.action_list[act_idx]
            prev_owner = self.world.owner(tgt)
            success = self.world.attack(atk, tgt)
            if success and prev_owner != self.world.owner(tgt):
                reward = 1.0

        # 2) check termination
        done = self.world.game_over()

        # 3) advance to next player
        self.world.next_player()

        # 4) rebuild legal-action list for the new player
        self.action_list = self._legal_attacks()
        self.action_space = gym.spaces.Discrete(max(1, len(self.action_list)))

        # if nobody can attack, end episode early
        if not self.action_list and not done:
            done = True
        if done:
            reward += 50.0

        return self._get_obs(), reward, done, False, {}

    # ----------------------------------------------------------
    # Render for debugging
    # ----------------------------------------------------------
    def render(self):
        print(self.world.get_map())
        print(self.world.last_event())
