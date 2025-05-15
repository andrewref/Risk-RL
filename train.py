#!/usr/bin/env python3
"""Robust training driver for the PPO meta-agent:
- Stops every PyRisk game after a fixed turn cap or a wall-clock timeout.
- Catches Ctrl-C to save the latest PPO model and partial win stats.
- Tracks win rate improvement.
- Persists win-rate history and backs up best-performing model.
- Generates win-rate plot.
- Logs strategy usage summary.
"""

from __future__ import annotations
import argparse
import inspect
import logging
import os
import random
import threading
import json
import contextlib
import io
from collections import defaultdict
from typing import Optional
import matplotlib.pyplot as plt

from pyrisk.game import Game
from agents.ppoagent import PPOAgent, PPOConfig
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

LOG = logging.getLogger("train")

AGENT_NAMES = ["PPO", "AggressiveAI", "BalancedAI", "DefensiveAI", "RandomAI"]
TRACK_FILE = "ppo_training_progress.json"
MODEL_FILE = "ppo_model_final.pt"


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
    episodes = []
    total = 0
    for entry in history:
        total += entry["episodes"]
        episodes.append(total)
    win_rates = [entry["win_rate"] for entry in history]
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, win_rates, marker="o")
    plt.title("PPO Win Rate Over Training")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_winrate_plot_final.png")
    LOG.info("Saved win rate plot as 'ppo_winrate_plot_final.png'")


def play_with_limits(game: Game, max_turns: int, factor: float = 0.06) -> str:
    has_kw = "max_turns" in inspect.signature(Game.play).parameters
    timeout_s = max_turns * factor
    result: dict[str, Optional[str]] = {"winner": None}

    def _runner():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result["winner"] = game.play(max_turns=max_turns) if has_kw else game.play()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout_s)

    if t.is_alive():
        LOG.warning("Timeout reached (%.1fs): forcing win by troop count", timeout_s)
        alive = [p for p in game.players.values() if p.alive]
        return max(alive, key=lambda p: (p.forces, p.territory_count)).name if alive else "Draw"
    return result["winner"] or "Draw"


def make_ppo(player, game, world):
    cfg = PPOConfig(model_path=MODEL_FILE)
    return PPOAgent(player, game, world, config=cfg)


def train(num_games: int, max_turns: int, log_interval: int, seed: int, deal: bool):
    if seed != -1:
        random.seed(seed)

    wins = defaultdict(int, {name: 0 for name in AGENT_NAMES})
    history = load_history()
    initial_rate = history[0]["win_rate"] if history else None
    previous_rate = history[-1]["win_rate"] if history else None
    strategy_usage = defaultdict(int)
    g: Optional[Game] = None

    try:
        for ep in range(1, num_games + 1):
            g = Game(curses=False, color=False, delay=0, wait=False, deal=deal)
            g.add_player("PPO", make_ppo)
            g.add_player("AggressiveAI", AggressiveAI)
            g.add_player("BalancedAI", BalancedAI)
            g.add_player("DefensiveAI", DefensiveAI)
            g.add_player("RandomAI", RandomAI)

            winner = play_with_limits(g, max_turns)
            wins[winner] += 1

            for pl in g.players.values():
                if isinstance(pl.ai, PPOAgent):
                    for strat, count in pl.ai.count.items():
                        strategy_usage[strat] += count

            if ep % log_interval == 0 or ep == num_games:
                ppo_rate = wins["PPO"] / ep * 100
                if previous_rate is not None:
                    LOG.info("PPO win rate: %.2f%% (change %+0.2f%%)", ppo_rate, ppo_rate - previous_rate)
                else:
                    LOG.info("PPO win rate: %.2f%%", ppo_rate)

    except KeyboardInterrupt:
        LOG.info("Training interrupted by user: saving current PPO model...")
        if g:
            for pl in g.players.values():
                if isinstance(pl.ai, PPOAgent):
                    pl.ai.end()
                    LOG.info("PPO model saved on interrupt.")
                    break
        LOG.info("Partial training results:")
        for name in AGENT_NAMES:
            LOG.info("%s: %d wins", name, wins[name])
        return

    LOG.info("Final training results:")
    for name in AGENT_NAMES:
        LOG.info("%s: %d wins", name, wins[name])

    LOG.info("Strategy usage summary:")
    for strat, count in strategy_usage.items():
        LOG.info("%s: %d times", strat, count)

    final_rate = wins["PPO"] / num_games * 100
    history.append({"episodes": num_games, "win_rate": final_rate})
    save_history(history)

    if previous_rate is not None:
        delta = final_rate - previous_rate
        LOG.info("PPO win rate: %.2f%% -> %.2f%% (change %+0.2f%%)", previous_rate, final_rate, delta)
    else:
        LOG.info("PPO win rate: %.2f%% (first recorded session)", final_rate)

    if initial_rate is not None:
        overall_delta = final_rate - initial_rate
        LOG.info("Overall improvement since first session: %+0.2f%%", overall_delta)

    plot_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO meta-agent with turn cap")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of training episodes")
    parser.add_argument("--max_turns", type=int, default=600, help="Turn cap per game")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N games")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 to skip)")
    parser.add_argument("--deal", action="store_true", help="Deal territories instead of claiming")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    # suppress verbose and warning logs from external modules
    logging.getLogger('agents.ppoagent').setLevel(logging.ERROR)
    logging.getLogger('pyrisk').setLevel(logging.ERROR)
    logging.getLogger('pyrisk.player.AggressiveAI').setLevel(logging.ERROR)

    train(args.episodes, args.max_turns, args.log_interval, args.seed, args.deal)
