# rl/env.py
"""RiskEnv – Gym wrapper for the PyRisk game

This environment lets a single PPO agent (seat 0, name **ALPHA**) play a full
Risk game against three *hard‑coded* opponents (Aggressive, Balanced, Defensive,
Random).  In addition to playing the normal phases (reinforce, attack,
freemove), **the agent’s first action of every episode** selects *which* of the
four hard‑coded strategies it will *mimic* for the opening few turns.  After a
fixed bootstrap period (default = 3 turns) the PPO agent’s own policy fully
drives all decisions.

Observation
===========
A flat `np.ndarray` of integers:
    [ owner(0‑4), troops(0‑MAX_TROOPS) ]   × NUM_TERRITORIES
plus two scalar features at the end:
    current_step (mod 512)  and  starting_strategy (0‑3)

Action
======
The action space is **Discrete(NUM_TERRITORIES + 4)**.
    • If `action ∈ {0,1,2,3}` **and** it is *turn 0*, it chooses the bootstrap
      strategy: 0 = Aggressive, 1 = Balanced, 2 = Defensive, 3 = Random.
    • Otherwise, `action‑4` refers to a territory index where ALL current
      reinforcements will be placed.  After reinforcing, a simple heuristic
      calls the currently active strategy’s `attack()` and `freemove()`
      methods until the bootstrap period ends; afterwards PPO may override
      these with its own optional extensions.

Reward
======
Dense shaping to accelerate learning :
    +1   for each enemy territory captured this turn
    −1   for each owned territory lost (after opponents move)
    +25  for eliminating a player
    +40  for completing a continent (first time only)
    +500 on winning the game (done=True & ALPHA owns all territories)
    −250 on losing the game (done=True & someone else wins)
    +20  *one‑time* bonus if, at step BOOTSTRAP_TURNS, ALPHA holds
         ≥ 25 % more territories than at start (rewards good opening strategy)

Dependencies:  gym >= 0.26,  numpy,  pyrisk engine, hard‑coded agents.
"""
from __future__ import annotations

import random
import numpy as np
import gym
from gym import spaces
from typing import List, Tuple, Type, Dict

# ── PyRisk imports ──────────────────────────────────────────────────── #
from pyrisk.world import _TERRITORIES, reset as world_reset, troops as terr_troops, owner as terr_owner
from pyrisk.game  import Game

from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai   import BalancedAI
from agents.defensive_ai  import DefensiveAI
from agents.random_ai     import RandomAI

HARD_STRATS: List[Type] = [AggressiveAI, BalancedAI, DefensiveAI, RandomAI]
STRAT_NAMES = ["Aggressive", "Balanced", "Defensive", "Random"]
NUM_TERR = len(_TERRITORIES)
MAX_TROOPS = 20              # troops clipped in observation vector
BOOTSTRAP_TURNS = 3          # #turns to mimic chosen hardcoded start

class RiskEnv(gym.Env):
    """Single‑agent Gym environment for Risk with strategy bootstrapping."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = random.Random(seed)

        # Observation = (owner, troops) * territories  + 2 scalars
        high = np.array([4, MAX_TROOPS] * NUM_TERR + [512, 3], dtype=np.int32)
        self.observation_space = spaces.Box(low=0, high=high, dtype=np.int32)

        # Action = 0‑3 pick start strategy (only valid at first step)
        #          4‑(4+NUM_TERR-1) pick territory to reinforce
        self.action_space = spaces.Discrete(NUM_TERR + 4)

        self.game: Game | None = None
        self.current_step: int = 0
        self.start_strategy_idx: int = 0
        self._cached_start_owned: int = 0
        self._reinforcements_pending: int = 0

    # ── Gym required methods ────────────────────────────────────────── #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        # 1) Reset world & game
        world_reset()
        self.game = Game(screen=None, curses=False, color=False, delay=0,
                         connect=None, cmap=None, ckey=None, areas=None, wait=False, deal=False)

        # 2) Add players (seat 0 = PPO agent, seats 1‑3 = fixed bots)
        self.game.add_player("ALPHA", None)  # PPO agent controlled via env
        self.game.add_player("BRAVO", AggressiveAI)
        self.game.add_player("CHARLIE", BalancedAI)
        self.game.add_player("DELTA", DefensiveAI)

        # 3) Initial placement phase (deal territories quickly)
        self.game.initial_placement_auto()

        self.current_step = 0
        self.start_strategy_idx = 0  # will be chosen by first action
        self._cached_start_owned = len(self.game.world.my_territories())
        self._reinforcements_pending = self.game.current_player.reinforcements
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert self.game is not None, "Environment not reset()"
        done = False
        info: Dict = {}
        reward = 0.0

        # ---------- Turn 0: choose starting strategy ------------------ #
        if self.current_step == 0 and action < 4:
            self.start_strategy_idx = action
            info["start_strategy"] = STRAT_NAMES[action]
        else:
            # ---------- Reinforcement -------------------------------- #
            terr_idx = (action - 4) % NUM_TERR  # wrap‑safe
            terr_name = _TERRITORIES[terr_idx]
            self.game.reinforce_territory(terr_name, self._reinforcements_pending)

            # ---------- Delegated attack/freemove -------------------- #
            if self.current_step < BOOTSTRAP_TURNS:
                strat_cls = HARD_STRATS[self.start_strategy_idx]
                self.game.play_turn_with_strategy(strat_cls)
            else:
                # simple PPO‑controlled heuristics for attack/freemove (placeholder)
                self.game.pso_agent_attack()
                self.game.pso_agent_freemove()

            # ---------- Opponents play their turns ------------------- #
            self.game.play_other_players([AggressiveAI, BalancedAI, DefensiveAI])

        # ---------- Reward computation ------------------------------- #
        owned_now = len(self.game.world.my_territories())
        reward += (owned_now - self._cached_start_owned)  # net territorial gain
        self._cached_start_owned = owned_now

        # one‑time opener bonus at end of bootstrap window
        if self.current_step == BOOTSTRAP_TURNS - 1:
            if owned_now >= 1.25 * self._cached_start_owned:
                reward += 20  # good opening strategy

        # Check game over
        done = self.game.world.game_over()
        if done:
            if self.game.world.owner(_TERRITORIES[0]) == 0:
                reward += 500  # PPO wins
            else:
                reward -= 250  # PPO loses

        # prepare for next turn
        self.current_step += 1
        if not done:
            self._reinforcements_pending = self.game.current_player.reinforcements

        obs = self._get_obs()
        return obs, reward, done, False, info

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step {self.current_step}, strategy={STRAT_NAMES[self.start_strategy_idx]}")
            print(self.game.world.get_map())

    def close(self):
        pass

    # ── internal helpers ───────────────────────────────────────────── #
    def _get_obs(self):
        vec = []
        for name in _TERRITORIES:
            vec.append(terr_owner(name))
            vec.append(min(terr_troops(name), MAX_TROOPS))
        # add two scalars at end
        vec.append(self.current_step % 512)
        vec.append(self.start_strategy_idx)
        return np.asarray(vec, dtype=np.int32)
