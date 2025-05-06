"""rl/env.py – full Risk game environment for ALPHA (seat 0)

▪ Spawns a new PyRisk game with three hard‑coded opponents.
▪ Allows a PPO agent to (optionally) pick an opening strategy for the
  first few turns, then take over fully.
▪ Provides dense shaping rewards so an end‑to‑end learner can still
  converge.

NOTE: This env is *also* imported by `strategy_selector_env.py`, so it
must be robust to partial / missing pieces inside `pyrisk`.
"""
from __future__ import annotations

import importlib, random, warnings, re
from typing import Dict, List, Type

import numpy as np
import gym
from gym import spaces

# ------------------------------------------------------------------- #
# 1.  Load pyrisk.world resiliently                                   #
# ------------------------------------------------------------------- #
pw = importlib.import_module("pyrisk.world")  # assumes package path

# --- world reset (fallback to no‑op) -------------------------------- #
world_reset = getattr(pw, "reset", getattr(pw, "reset_world", lambda *a, **k: None))
if world_reset.__name__ == "<lambda>":
    warnings.warn("pyrisk.world lacks reset(); using no‑op fallback")

# --- troops / owner helpers with graceful degradation --------------- #
if hasattr(pw, "troops"):
    terr_troops = pw.troops                # type: ignore
elif hasattr(pw, "_TROOPS"):
    terr_troops = lambda n: pw._TROOPS.get(n, 1)  # type: ignore
else:
    terr_troops = lambda _n: 1

if hasattr(pw, "owner"):
    terr_owner = pw.owner                  # type: ignore
elif hasattr(pw, "_OWNER"):
    terr_owner = lambda n: pw._OWNER.get(n, -1)   # type: ignore
else:
    terr_owner = lambda _n: -1

# --- derive territory list ----------------------------------------- #
if hasattr(pw, "_TERRITORIES"):
    _TERRITORIES = list(pw._TERRITORIES)           # type: ignore
elif hasattr(pw, "World"):
    _TERRITORIES = list(pw.World().territories)    # type: ignore
elif hasattr(pw, "CONNECT"):
    terrs: set[str] = set()
    for line in pw.CONNECT.strip().splitlines():
        terrs.update(t.strip() for t in re.split(r"--", line) if t.strip())
    _TERRITORIES = sorted(terrs)
    warnings.warn("_TERRITORIES missing – parsed list from CONNECT text")
else:
    raise ImportError("Could not derive territory list from pyrisk.world")

# ------------------------------------------------------------------- #
# 2.  Remaining imports & constants                                   #
# ------------------------------------------------------------------- #
from pyrisk.game import Game
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai   import BalancedAI
from agents.defensive_ai  import DefensiveAI
from agents.random_ai     import RandomAI

HARD_STRATS: List[Type] = [AggressiveAI, BalancedAI, DefensiveAI, RandomAI]
STRAT_NAMES              = ["Aggressive", "Balanced", "Defensive", "Random"]
NUM_TERR                 = len(_TERRITORIES)
MAX_TROOPS               = 20
BOOTSTRAP_TURNS          = 3

# Create a placeholder Human AI class for the ALPHA player
class HumanControlledAI:
    def __init__(self, player, game, world, **kwargs):
        self.player = player
        self.game = game
        self.world = world
        
    def reinforce(self):
        # Will be controlled by the RL agent
        return None
        
    def attack(self):
        # Will be controlled by the RL agent
        return None
        
    def freemove(self):
        # Will be controlled by the RL agent
        return None

# ------------------------------------------------------------------- #
#  Activate helper monkey‑patches (play_turn_with_strategy etc.)       #
# ------------------------------------------------------------------- #
import rl.strategies  # noqa: F401  (import solely for side‑effects)

# ------------------------------------------------------------------- #
# 3.  Helper function to get territories owned by a player           #
# ------------------------------------------------------------------- #
def get_owned_territories(world, player_idx=0):
    """Get territories owned by the specified player index."""
    # Try different approaches to find territories owned by the player
    try:
        # First try: check if there's a method for this
        if hasattr(world, "my_territories") and callable(world.my_territories):
            return world.my_territories()
        # Second try: check if there's a method with player index
        elif hasattr(world, "player_territories") and callable(world.player_territories):
            return world.player_territories(player_idx)
        # Third try: filter territories by owner
        else:
            return [t for t in world.territories if terr_owner(t) == player_idx]
    except Exception as e:
        warnings.warn(f"Error getting owned territories: {e}")
        return []  # Return empty list as fallback

# ------------------------------------------------------------------- #
# 4.  Gym Environment                                                 #
# ------------------------------------------------------------------- #
class RiskEnv(gym.Env):
    """Single‑agent PPO environment: ALPHA vs three rule‑based bots."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = random.Random(seed)

        high = np.array([4, MAX_TROOPS] * NUM_TERR + [512, 3], dtype=np.int32)
        self.observation_space = spaces.Box(low=0, high=high, dtype=np.int32)
        self.action_space      = spaces.Discrete(NUM_TERR + 4)

        self.game: Game | None = None
        self.current_step      = 0
        self.start_strategy_idx= 0
        self._cached_owned     = 0
        self._reinforcements   = 0
        self.strategy_counts   = [0]*4

    # --------------------------- Gym API --------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        world_reset()

        # Get default map data from world module
        connect = getattr(pw, "CONNECT", None)
        areas = getattr(pw, "AREAS", None)
        
        # create Game – tolerate minimal constructor if full one fails
        try:
            self.game = Game(screen=None, curses=False, color=False, delay=0,
                            connect=connect, cmap=None, ckey=None, areas=areas,
                            wait=False, deal=False)
        except TypeError:
            try:
                # Try with minimal args
                self.game = Game()
            except Exception as e:
                # If all else fails, create a very minimal Game with required attributes
                self.game = Game.__new__(Game)
                self.game.world = pw.World()
                self.game.players = []
                self.game.current_player = None
                warnings.warn(f"Failed to create Game properly: {e}")

        # add players - use HumanControlledAI for ALPHA instead of None
        self.game.add_player("ALPHA", HumanControlledAI)
        self.game.add_player("BRAVO", AggressiveAI)
        self.game.add_player("CHARLIE", BalancedAI)
        self.game.add_player("DELTA", DefensiveAI)

        # initial placement (random deal)
        if hasattr(self.game, "initial_placement_auto"):
            try:
                self.game.initial_placement_auto()
            except Exception as e:
                warnings.warn(f"initial_placement_auto failed: {e}")
        # else: assume world_reset handled territory assignment

        self.current_step       = 0
        self.start_strategy_idx = 0
        
        # Use our helper function instead of directly calling my_territories()
        self._cached_owned = len(get_owned_territories(self.game.world, 0))
        self._reinforcements = getattr(self.game.current_player, "reinforcements", 3)
        return self._get_obs(), {}

    # ---------------------------------------------------------------- #
    def step(self, action: int):
        assert self.game is not None, "Call reset() first"
        info: Dict = {}
        reward     = 0.0

        # ---- choose starting strategy on first step ---------------- #
        if self.current_step == 0 and action < 4:
            self.start_strategy_idx          = action
            self.strategy_counts[action]    += 1
            info["start_strategy"]           = STRAT_NAMES[action]
        else:
            terr_name = _TERRITORIES[(action - 4) % NUM_TERR]
            self.game.reinforce_territory(terr_name, self._reinforcements)

            if self.current_step < BOOTSTRAP_TURNS:
                self.game.play_turn_with_strategy(HARD_STRATS[self.start_strategy_idx])
            else:
                self.game.pso_agent_attack(); self.game.pso_agent_freemove()

            self.game.play_other_players([AggressiveAI, BalancedAI, DefensiveAI])

        # ---- reward shaping --------------------------------------- #
        owned_now = len(get_owned_territories(self.game.world, 0))
        reward   += (owned_now - self._cached_owned)
        if self.current_step == BOOTSTRAP_TURNS-1 and owned_now >= 1.25*self._cached_owned:
            reward += 20
        self._cached_owned = owned_now

        done = self.game.world.game_over()
        if done:
            winner_is_alpha = self.game.world.owner(_TERRITORIES[0]) == 0
            reward += 500 if winner_is_alpha else -250
            info["strategy_counts"] = {n: self.strategy_counts[i] for i,n in enumerate(STRAT_NAMES)}
        else:
            self._reinforcements = getattr(self.game.current_player, "reinforcements", 3)

        self.current_step += 1
        return self._get_obs(), reward, done, False, info

    # --------------------------- Helpers --------------------------- #
    def _get_obs(self):
        vec = []
        for terr in _TERRITORIES:
            vec.extend((terr_owner(terr), min(terr_troops(terr), MAX_TROOPS)))
        vec.extend((self.current_step % 512, self.start_strategy_idx))
        return np.asarray(vec, dtype=np.int32)

    def render(self, mode="human"):
        if mode == "human":
            strat = STRAT_NAMES[self.start_strategy_idx]
            print(f"Step {self.current_step} | strategy={strat}")
            print(self.game.world.get_map())

    def close(self):
        if self.game and hasattr(self.game, "close"):
            self.game.close()