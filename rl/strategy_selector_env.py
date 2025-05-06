"""rl/strategy_selector_env.py

A minimal Gym environment where the **only action** is to pick one of the
4 hard‑coded opening strategies (Aggressive, Balanced, Defensive, Random).
The environment then runs **7 full turns** of a fresh Risk game using that
strategy and returns a reward:  +1 if ALPHA owns more territories than it
started with, otherwise −1.  Each episode therefore consists of a single
action/decision, making it ideal for a lightweight RL/bandit learner.

This file depends on `rl.env.RiskEnv`, which provides the full Risk game
logic and strategy plumbing.
"""
from __future__ import annotations
import warnings

from typing import Dict, List, Type
import numpy as np
import gym
from gym import spaces
import random
from rl.env import get_owned_territories

from rl.env import RiskEnv, HARD_STRATS, STRAT_NAMES
from rl.env import RiskEnv, HARD_STRATS, STRAT_NAMES, _TERRITORIES  # ✅ correct and safe


NUM_STRATS     = len(HARD_STRATS)     # 4
SEGMENT_TURNS  = 7                    # play 7 turns per episode

class StrategySelectorEnv(gym.Env):
    """Meta‑environment: choose a start strategy, get reward after 7 turns."""

    metadata = {"render.modes": ["human"]}

    # ---------------------------------------------------------------- #
    def __init__(self):
        super().__init__()
        tmp_env = RiskEnv()
        self.observation_space = tmp_env.observation_space
        self.action_space      = spaces.Discrete(NUM_STRATS)  # 0‑3

        self.inner_env: RiskEnv | None = None
        self.start_obs: np.ndarray | None = None
        self.current_strategy: int = -1

    # ---------------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.inner_env = RiskEnv()
        self.start_obs, _ = self.inner_env.reset()
        self.current_strategy = -1
        return self.start_obs.copy(), {}

    # ---------------------------------------------------------------- #
    def step(self, action: int):
        assert self.inner_env is not None, "reset() must be called first"
        self.current_strategy = int(action) % NUM_STRATS
        self.inner_env.start_strategy_idx = self.current_strategy

        # ------------------------------------------------------------- #
        # Play SEGMENT_TURNS turns with that strategy
        # ------------------------------------------------------------- #
        for turn in range(SEGMENT_TURNS):
            # Reinforce the first ALPHA‑owned territory
            owned_territories = get_owned_territories(self.inner_env.game.world, 0)
            if not owned_territories:
                warnings.warn("ALPHA owns no territories. Skipping step.")
                break  # or: continue  # depending on what behavior you prefer

            alpha_terr = owned_territories[0]

            terr_idx  = _TERRITORIES.index(alpha_terr)
            action_id = 4 + terr_idx  # territory actions are offset by 4

            # Step the inner env once
            self.inner_env.step(action_id)
            if self.inner_env.game.world.game_over():
                break


        # ------------------------------------------------------------- #
        # Compute reward
        # ------------------------------------------------------------- #
        owned_before = sum(1 for i in range(0, len(self.start_obs), 2) if self.start_obs[i] == 0)
        owned_after = len(get_owned_territories(self.inner_env.game.world, player_idx=0))
        reward       = 1.0 if owned_after > owned_before else -1.0

        info: Dict = {
            "strategy": STRAT_NAMES[self.current_strategy],
            "owned_before": owned_before,
            "owned_after": owned_after,
        }
        done = True  # single decision per episode
        obs  = self.start_obs.copy()
        return obs, reward, done, False, info

    # ---------------------------------------------------------------- #
    def render(self, mode="human"):
        if mode == "human":
            strat = STRAT_NAMES[self.current_strategy] if self.current_strategy >= 0 else "N/A"
            print(f"Selector chose: {strat}")

    def close(self):
        if self.inner_env:
            self.inner_env.close()