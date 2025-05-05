# agents/ppo.py
"""PpoAI – wrapper that lets a *trained* Stable‑Baselines3 PPO model play
in the classic PyRisk ncurses UI exactly like a hard‑coded bot.

Usage (after you have a model):
    python pyrisk/pyrisk.py PpoAI AggressiveAI BalancedAI DefensiveAI

Assumptions
-----------
* The PPO model was trained with `rl/env.py` and saved to
  `checkpoints/ppo_risk_final.zip`.
* The model’s action space is **Discrete(NUM_TERRITORIES + 4)** as defined in
  `RiskEnv`.
* We only use the model for the **reinforcement phase** for now; attack and
  freemove are no‑ops (you can extend them later).
"""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from pyrisk.world import _TERRITORIES, troops as terr_troops, owner as terr_owner
from typing import Dict

MODEL_PATH = "checkpoints/ppo_risk_final.zip"

class PpoAI:
    """Bridge class so the training model can play in pyrisk CLI."""

    def __init__(self, player, game, world, **kwargs):
        self.player = player
        self.game   = game
        self.world  = world
        self.model  = PPO.load(MODEL_PATH)
        self.step   = 0
        self.start_strategy_idx = 0  # recorded from first action if <4

    # ----------------------------------------------------------------- #
    #  Required hooks                                                   #
    # ----------------------------------------------------------------- #
    def start(self):
        pass

    def end(self):
        pass

    def event(self, msg):
        pass

    # ----------------------------------------------------------------- #
    #  Initial placement – keep simple / random for now                 #
    # ----------------------------------------------------------------- #
    def initial_placement(self, empty_list, remaining):
        if empty_list:
            # just pick first unowned – training didn’t optimize this phase
            return empty_list[0].name
        # extra troops after every territory is owned → call reinforce()
        return self.reinforce(1).popitem()[0]

    # ----------------------------------------------------------------- #
    #  Reinforcement – main phase driven by PPO model                   #
    # ----------------------------------------------------------------- #
    def reinforce(self, troops: int) -> Dict[str, int]:
        obs = self._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)

        if self.step == 0 and action < 4:
            # strategy index – just record for info; gameplay unaffected here
            self.start_strategy_idx = action
            # choose a fallback territory
            terr_name = _TERRITORIES[0]
        else:
            terr_idx  = (action - 4) % len(_TERRITORIES)
            terr_name = _TERRITORIES[terr_idx]

        self.step += 1  # increment internal step counter
        return {terr_name: troops}

    # ----------------------------------------------------------------- #
    #  Attack / Freemove – no‑ops for now                               #
    # ----------------------------------------------------------------- #
    def attack(self):
        return []

    def freemove(self):
        return None

    # ----------------------------------------------------------------- #
    #  Helper: build observation identical to RiskEnv._get_obs()        #
    # ----------------------------------------------------------------- #
    def _get_obs(self):
        vec = []
        for name in _TERRITORIES:
            vec.append(terr_owner(name))
            vec.append(min(terr_troops(name), 20))
        vec.append(self.step % 512)            # dummy step scalar
        vec.append(self.start_strategy_idx)    # starting strategy scalar
        return np.asarray(vec, dtype=np.int32)
