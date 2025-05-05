# rl/strategies.py
"""Utility glue so `RiskEnv` can easily invoke a hard‑coded strategy inside
an ongoing Game turn.

It monkey‑patches three helper methods onto the `pyrisk.game.Game` class:

    • `play_turn_with_strategy(strat_cls)` – runs reinforce / attack /
      freemove for the *current player* using the given AI class.
    • `pso_agent_attack()` and `pso_agent_freemove()` – very small placeholders
      that do nothing (you can later replace with PPO‑driven logic if you wish).
    • `play_other_players(strat_list)` – cycles each non‑RL player for one full
      turn using its designated strategy class.

This keeps `rl/env.py` clean.
"""
from __future__ import annotations

from typing import Type, List
from pyrisk.game import Game

# ---- Patch helpers onto Game at import time ------------------------- #

def _play_turn_with_strategy(self: Game, strat_cls: Type):
    """Run a full turn (reinforce, attack, freemove) for the *current player*
    using the given hard‑coded strategy class.
    Assumes `self.current_player` is the player whose turn it is.
    """
    player = self.current_player
    ai     = strat_cls(player, self, self.world)

    # --- Reinforce phase (drop all troops on ai.initial choice) ------- #
    troops = player.reinforcements
    terr   = ai.reinforce(troops)
    if isinstance(terr, dict):           # most bots return dict
        self.reinforce_bulk(terr)
    else:                                # fallback single territory
        self.reinforce_territory(terr, troops)

    # --- Attack phase ------------------------------------------------- #
    for src, tgt, cont_fn, move_fn in ai.attack():
        self.resolve_attack(src, tgt, cont_fn, move_fn)

    # --- Freemove phase ---------------------------------------------- #
    fm = ai.freemove()
    if fm:
        self.apply_freemove(*fm)


def _play_other_players(self: Game, strat_list: List[Type]):
    """Iterate the rest of the round so that each non‑RL seat takes a turn
    using its associated strategy class.
    strat_list must be indexed the same as player seats 1..N.
    """
    for seat in range(1, len(self.players)):
        self.world._CUR = seat  # advance to that player manually
        strat_cls = strat_list[seat-1]
        self.play_turn_with_strategy(strat_cls)

        # finish phase bookkeeping if game not over
        if self.world.game_over():
            break


def _noop(self: Game):
    """Placeholder – PPO agent can learn to override later."""
    pass

# ---- Monkey‑patch --------------------------------------------------- #
Game.play_turn_with_strategy = _play_turn_with_strategy           # type: ignore
Game.play_other_players      = _play_other_players                # type: ignore
Game.pso_agent_attack        = _noop                              # type: ignore
Game.pso_agent_freemove      = _noop                              # type: ignore

__all__ = [
    "_play_turn_with_strategy",
    "_play_other_players",
]
