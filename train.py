#!/usr/bin/env python3
"""Robust training driver for the PPO meta-agent.

- Stops every PyRisk game after a fixed turn cap (now 250 turns) or a wall-clock timeout.
- Catches Ctrl-C to save the latest PPO model and partial win stats.
- Tracks win rate improvement.
- Persists win-rate history and backs up best-performing model.
- Generates win-rate plot.
- Logs strategy usage summary.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import inspect
import json
import logging
import os
import random
import threading
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt

from pyrisk.game import Game
from AI.ppoagent import PPOAgent, PPOConfig
from AI.aggressive_ai import AggressiveAI
from AI.balanced_ai import BalancedAI
from AI.defensive_ai import DefensiveAI
from AI.random_ai import RandomAI
from AI.better_ai import BetterAI

LOG = logging.getLogger("train")
AGENT_NAMES = ["PPO", "AggressiveAI", "BalancedAI", "DefensiveAI", "RandomAI", "BetterAI"]

TRACK_FILE = "ppo_training_progress.json"
MODEL_FILE = "ppo_model_final.pt"


def load_history() -> list[dict]:
    """Load training history from disk, ensuring a list of entries."""
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
    return []


def save_history(history: list[dict]) -> None:
    """Save training history to disk."""
    with open(TRACK_FILE, "w") as f:
        json.dump(history, f, indent=2)


def plot_history(history: list[dict]) -> None:
    """Generate and save a plot of win rate over training games."""
    if not history:
        return

    total = 0
    games_cumulative = []
    win_rates = []

    for entry in history:
        total += entry.get("episodes", 0)
        games_cumulative.append(total)
        win_rates.append(entry.get("win_rate", 0.0))

    plt.figure(figsize=(8, 5))
    plt.plot(games_cumulative, win_rates, marker="o")
    plt.title("PPO Win Rate Over Training")
    plt.xlabel("Games")  
    plt.ylabel("Win Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_winrate_plot_final2.png")
    LOG.info("Saved win rate plot as 'ppo_winrate_plot_final2.png'")


def play_with_limits(game: Game, max_turns: int, factor: float = 0.06) -> str:
    """Run a single game, enforcing a turn cap and a wall-clock timeout."""
    sig = inspect.signature(game.play)
    has_kw = "max_turns" in sig.parameters
    timeout_s = max_turns * factor

    result: dict[str, Optional[str]] = {"winner": None}

    def _runner():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if has_kw:
                result["winner"] = game.play(max_turns=max_turns)
            else:
                result["winner"] = game.play()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        LOG.warning("Timeout reached (%.1fs): forcing win by troop count", timeout_s)
        alive_players = [p for p in game.players.values() if p.alive]
        if alive_players:
            return max(alive_players, key=lambda p: (p.forces, p.territory_count)).name
        return "Draw"

    return result["winner"] or "Draw"


def make_ppo(player, game, world):
    """Factory for creating a PPOAgent with default configuration."""
    cfg = PPOConfig(model_path=MODEL_FILE)
    return PPOAgent(player, game, world, config=cfg)


def train(num_games: int, max_turns: int, log_interval: int, seed: int, deal: bool) -> None:
    """Main training loop for the PPO meta-agent."""
    if seed != -1:
        random.seed(seed)

    wins = defaultdict(int, {name: 0 for name in AGENT_NAMES})
    strategy_usage = defaultdict(int)

    history = load_history()
    initial_rate = history[0].get("win_rate") if history else None
    previous_rate = history[-1].get("win_rate") if history else None

    current_game: Optional[Game] = None

    try:
        for episode in range(1, num_games + 1):
            current_game = Game(curses=False, color=False, delay=0, wait=False, deal=deal)
            current_game.add_player("PPO", make_ppo)
#            current_game.add_player("AggressiveAI", AggressiveAI)
#            current_game.add_player("BalancedAI", BalancedAI)
 #           current_game.add_player("DefensiveAI", DefensiveAI)
  #          current_game.add_player("RandomAI", RandomAI)
            current_game.add_player("BetterAI", BetterAI)
            winner = play_with_limits(current_game, max_turns)
            wins[winner] += 1

            for pl in current_game.players.values():
                if isinstance(pl.ai, PPOAgent):
                    for strat, count in pl.ai.count.items():
                        strategy_usage[strat] += count

            if episode % log_interval == 0 or episode == num_games:
                ppo_rate = wins["PPO"] / episode * 100
                if previous_rate is not None:
                    LOG.info("PPO win rate: %.2f%% (change %+0.2f%%)", ppo_rate, ppo_rate - previous_rate)
                else:
                    LOG.info("PPO win rate: %.2f%%", ppo_rate)

    except KeyboardInterrupt:
        LOG.info("Training interrupted by user: saving current PPO model...")
        if current_game:
            for pl in current_game.players.values():
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
        LOG.info("Overall improvement since first recorded session: %+0.2f%%", overall_delta)

    plot_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO meta-agent with turn cap")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max_turns", type=int, default=250, help="Turn cap per game (now 250)")  # <-- updated
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N games")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 to skip)")
    parser.add_argument("--deal", action="store_true", help="Deal territories instead of claiming")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("agents.ppoagent").setLevel(logging.ERROR)
    logging.getLogger("pyrisk").setLevel(logging.ERROR)
    logging.getLogger("pyrisk.player.AggressiveAI").setLevel(logging.ERROR)

    train(
        num_games=args.episodes,
        max_turns=args.max_turns,
        log_interval=args.log_interval,
        seed=args.seed,
        deal=args.deal,
    )
