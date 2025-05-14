#!/usr/bin/env python3
"""Robust **training driver** for the PPO meta‑agent
===================================================
Stops every PyRisk game after a fixed *turn cap* **or** a wall‑clock timeout.
Catches Ctrl‑C to save the latest PPO model and partial win stats.
Also tracks win rate improvement to estimate learning progress.
Persists full win rate history between training sessions.
Backs up the best-performing model.
Generates win-rate plot over time.
Logs strategy usage summary for PPO.
"""
from __future__ import annotations
import argparse
import inspect
import logging
import os
import random
import threading
import json
import shutil
from collections import defaultdict
from typing import Optional
import matplotlib.pyplot as plt

from pyrisk.game import Game
from agents.ppoagent import PPOAgent
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

LOG = logging.getLogger("train")

AGENT_NAMES = [
    "PPO",
    "AggressiveAI",
    "BalancedAI",
    "DefensiveAI",
    "RandomAI",
]

TRACK_FILE = "ppo_training_progress.json"
MODEL_FILE = "ppo_model.pt"
BEST_MODEL_FILE = "ppo_model_best.pt"


def load_history() -> list[dict]:
    if os.path.exists(TRACK_FILE):
        try:
            with open(TRACK_FILE, "r") as f:
                data = json.load(f)
                return data.get("history", [])
        except Exception:
            return []
    return []


def save_history(history: list[dict]):
    with open(TRACK_FILE, "w") as f:
        json.dump({"history": history}, f)


def plot_history(history: list[dict]):
    if not history:
        return
    episodes = [sum(d["episodes"] for d in history[:i + 1]) for i in range(len(history))]
    win_rates = [d["win_rate"] for d in history]
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, win_rates, marker="o", color="blue")
    plt.title("PPO Win Rate Over Training")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_winrate_plot.png")
    LOG.info("📊 Saved win rate plot as 'ppo_winrate_plot.png'")


def play_with_limits(game: Game, max_turns: int, factor: float = 0.06) -> str:
    has_kw = "max_turns" in inspect.signature(Game.play).parameters
    timeout_s = max_turns * factor
    result: dict[str, Optional[str]] = {"winner": None}

    def _runner():
        result["winner"] = game.play(max_turns=max_turns) if has_kw else game.play()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout_s)

    if t.is_alive():
        LOG.warning("⚠️  Timeout (%.1fs) reached – forcing win by troop count", timeout_s)
        alive = [p for p in game.players.values() if p.alive]
        return max(alive, key=lambda p: (p.forces, p.territory_count)).name if alive else "Draw"

    return result["winner"] or "Draw"


def train(num_games: int, max_turns: int, log_interval: int, seed: int, deal: bool):
    if seed != -1:
        random.seed(seed)

    wins = defaultdict(int, {name: 0 for name in AGENT_NAMES})
    g: Optional[Game] = None
    history = load_history()
    initial_rate = history[0]['win_rate'] if history else None
    previous_rate = history[-1]['win_rate'] if history else None
    strategy_usage = defaultdict(int)

    try:
        for ep in range(1, num_games + 1):
            g = Game(curses=False, color=False, delay=0, wait=False, deal=deal)
            ppo_agent = PPOAgent
            g.add_player("PPO",          ppo_agent)
            g.add_player("AggressiveAI", AggressiveAI)
            g.add_player("BalancedAI",   BalancedAI)
            g.add_player("DefensiveAI",  DefensiveAI)
            g.add_player("RandomAI",     RandomAI)

            winner = play_with_limits(g, max_turns)
            wins[winner] += 1

            for pl in g.players.values():
                if isinstance(pl.ai, PPOAgent):
                    for k, v in pl.ai.count.items():
                        strategy_usage[k] += v

            if ep % log_interval == 0 or ep == num_games:
                LOG.info("Game %d/%d – wins so far %s", ep, num_games, dict(wins))
                ppo_rate = wins['PPO'] / ep * 100
                if previous_rate is not None:
                    LOG.info("📈 PPO win rate: %.2f%% (Change: %+0.2f%%)", ppo_rate, ppo_rate - previous_rate)
                else:
                    LOG.info("📈 PPO win rate: %.2f%%", ppo_rate)

    except KeyboardInterrupt:
        LOG.warning("⛔ Training interrupted by user – saving current PPO model …")
        if g:
            for pl in g.players.values():
                if isinstance(pl.ai, PPOAgent):
                    pl.ai.end()
                    LOG.info("✅ PPO model saved on interrupt.")
                    break
        logging.getLogger('pyrisk').setLevel(logging.WARNING)
        LOG.info("\n📊 Partial Training Results:")
        for name in AGENT_NAMES:
            print(f"{name}: {wins[name]} wins")
        return

    logging.getLogger('pyrisk').setLevel(logging.WARNING)
    LOG.info("\n📊 Final Training Results:")
    for name in AGENT_NAMES:
        print(f"{name}: {wins[name]} wins")

    LOG.info("\n📊 PPO Strategy Usage Summary:")
    for strat, count in strategy_usage.items():
        print(f"{strat:>10}: {count:3d} times")

    final_rate = wins['PPO'] / num_games * 100
    history.append({"episodes": num_games, "win_rate": final_rate})
    save_history(history)

    if previous_rate is not None:
        delta = final_rate - previous_rate
        LOG.info("\n📈 PPO Win Rate: %.2f%% → %.2f%% (Change: %+0.2f%%)", previous_rate, final_rate, delta)
        if final_rate > previous_rate and os.path.exists(MODEL_FILE):
            shutil.copy(MODEL_FILE, BEST_MODEL_FILE)
            LOG.info("💾 Best PPO model backed up as '%s'", BEST_MODEL_FILE)
    else:
        LOG.info("\n📈 PPO Win Rate: %.2f%% (First recorded session)", final_rate)

    if initial_rate is not None:
        overall_delta = final_rate - initial_rate
        LOG.info("📊 Overall Improvement since first session: %+0.2f%%", overall_delta)

    plot_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train PPO meta‑agent with safe turn cap")
    parser.add_argument("--episodes",     type=int, default=200,
                        help="Number of training episodes (games)")
    parser.add_argument("--max_turns",    type=int, default=600,
                        help="Turn cap per game; games exceeding this will be forced to end")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log progress after this many games")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for reproducibility (-1 for no seeding)")
    parser.add_argument("--deal",         action="store_true",
                        help="Deal territories instead of manual claiming")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")
    train(args.episodes, args.max_turns, args.log_interval, args.seed, args.deal)
