#!/usr/bin/env python3
"""Evaluate the trained PPO meta‑agent with the **same bullet‑proof turn‑cap
logic** used in train.py.

* First, if `Game.play(max_turns=…)` exists we simply pass the cap.
* Otherwise we run `Game.play()` in a **daemon thread** with a wall‑clock
  timeout proportional to `max_turns` (turns × 0.06 s).  When the timeout hits
  we end the game by troop‑count → territory‑count tiebreak.
* No monkey‑patching, no single‑turn discovery – evaluation can never hang.
"""
from __future__ import annotations
import argparse
import inspect
import logging
import random
import threading
from collections import defaultdict
from typing import Optional

from pyrisk.game import Game
from agents.ppoagent import PPOAgent
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

LOG = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# Fallback: run game.play in a thread and kill after timeout
# ---------------------------------------------------------------------------

def _play_with_timeout(game: Game, max_turns: int, factor: float = 0.06) -> str:
    """Run `game.play()` but give up after `max_turns × factor` seconds."""
    timeout_s: float = max(10.0, max_turns * factor)
    result: dict[str, Optional[str]] = {"winner": None}

    def _runner():
        result["winner"] = game.play()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        LOG.warning("⚠️  Timeout (%ss) reached – forcing win by troop count", round(timeout_s,1))
        alive = [p for p in game.players.values() if p.alive]
        return max(alive, key=lambda p: (p.forces, p.territory_count)).name if alive else "Draw"
    return result["winner"] or "Draw"

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(num_games: int, max_turns: int, seed: int, deal: bool, log_interval: int):
    if seed != -1:
        random.seed(seed)

    wins = defaultdict(int)

    for ep in range(1, num_games + 1):
        g = Game(curses=False, color=False, delay=0, wait=False, deal=deal)
        g.add_player("PPO",          PPOAgent)
        g.add_player("AggressiveAI", AggressiveAI)
        g.add_player("BalancedAI",   BalancedAI)
        g.add_player("DefensiveAI",  DefensiveAI)
        g.add_player("RandomAI",     RandomAI)

        if "max_turns" in inspect.signature(Game.play).parameters:
            winner = g.play(max_turns=max_turns)
        else:
            winner = _play_with_timeout(g, max_turns)

        wins[winner] += 1

        if ep % log_interval == 0 or ep == num_games:
            LOG.info("Game %d/%d – wins so far %s", ep, num_games, dict(wins))

    LOG.info("\n📊 Final Evaluation Results:")
    for k, v in wins.items():
        print(f"{k}: {v} wins")
    return dict(wins)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate PPO meta‑agent with turn cap")
    p.add_argument("--games",      type=int, default=50, help="Number of evaluation games")
    p.add_argument("--max_turns",  type=int, default=600, help="Turn cap per game")
    p.add_argument("--seed",       type=int, default=42,  help="Random seed (‑1 = no seeding)")
    p.add_argument("--deal",       action="store_true",  help="Deal territories instead of claiming")
    p.add_argument("--log_interval", type=int, default=10, help="Progress log frequency")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    evaluate(args.games, args.max_turns, args.seed, args.deal, args.log_interval)
