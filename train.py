#!/usr/bin/env python3
"""Robust **training driver** for the PPO meta‑agent
===================================================
Stops every PyRisk game after a fixed *turn cap* **or** a wall‑clock timeout.
Catches Ctrl‑C to save the latest PPO model and partial win stats.
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
from agents.balanced_ai    import BalancedAI
from agents.defensive_ai   import DefensiveAI
from agents.random_ai      import RandomAI

LOG = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Agent names for summary
# ---------------------------------------------------------------------------
AGENT_NAMES = [
    "PPO",
    "AggressiveAI",
    "BalancedAI",
    "DefensiveAI",
    "RandomAI",
]

# ---------------------------------------------------------------------------
# Helper: run a Game with both turn‑cap and wall‑time limit
# ---------------------------------------------------------------------------

def play_with_limits(game: Game, max_turns: int, factor: float = 0.06) -> str:
    """Run ``game.play()`` in a daemon thread; abort after *max_turns × factor* s.

    * If ``Game.play`` accepts ``max_turns`` we pass it directly.
    * If the timeout hits we end the match by (troops → territories).
    """
    has_kw = "max_turns" in inspect.signature(Game.play).parameters
    timeout_s: float = max_turns * factor
    result: dict[str, Optional[str]] = {"winner": None}

    def _runner():
        result["winner"] = game.play(max_turns=max_turns) if has_kw else game.play()

    t = threading.Thread(target=_runner, daemon=True)
    t.start(); t.join(timeout_s)

    if t.is_alive():
        LOG.warning("⚠️  Timeout (%.1fs) reached – forcing win by troop count", timeout_s)
        alive = [p for p in game.players.values() if p.alive]
        if not alive:
            return "Draw"
        return max(alive, key=lambda p: (p.forces, p.territory_count)).name

    return result["winner"] or "Draw"

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(num_games: int, max_turns: int, log_interval: int, seed: int, deal: bool):
    if seed != -1:
        random.seed(seed)

    wins = defaultdict(int, {name: 0 for name in AGENT_NAMES})
    g = None
    try:
        for ep in range(1, num_games + 1):
            g = Game(curses=False, color=False, delay=0, wait=False, deal=deal)
            g.add_player("PPO",          PPOAgent)
            g.add_player("AggressiveAI", AggressiveAI)
            g.add_player("BalancedAI",   BalancedAI)
            g.add_player("DefensiveAI",  DefensiveAI)
            g.add_player("RandomAI",     RandomAI)

            winner = play_with_limits(g, max_turns)
            wins[winner] += 1

            if ep % log_interval == 0 or ep == num_games:
                LOG.info("Game %d/%d – wins so far %s", ep, num_games, dict(wins))

    except KeyboardInterrupt:
        LOG.warning("⛔ Training interrupted by user – saving current PPO model …")
        # Save PPO model from current game if available
        if g:
            for pl in g.players.values():
                if isinstance(pl.ai, PPOAgent):
                    pl.ai.end()  # flush PPO updates and save
                    LOG.info("✅ PPO model saved on interrupt.")
                    break
        # Partial summary
        LOG.info("\n📊 Partial Training Results:")
        for name in AGENT_NAMES:
            print(f"{name}: {wins[name]} wins")
        return

    # Final summary
    LOG.info("\n📊 Final Training Results:")
    for name in AGENT_NAMES:
        print(f"{name}: {wins[name]} wins")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser("Train PPO meta‑agent with safe turn cap")
    p.add_argument("--episodes",     type=int, default=100)
    p.add_argument("--max_turns",    type=int, default=600)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed",         type=int, default=42, help="‑1 = no seeding")
    p.add_argument("--deal",         action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")
    train(args.episodes, args.max_turns, args.log_interval, args.seed, args.deal)