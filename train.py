#!/usr/bin/env python3
"""
Robust training driver for the PPO meta-agent.

Adds CSV logs for:
1. Episode reward curve
2. Strategy-usage frequency
3. Strategy-switch timing
4. + NEW: critic / actor diagnostic plots
"""
from __future__ import annotations
import argparse, contextlib, io, inspect, json, logging, os, random, threading
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np                    # ← NEW (for plotting arrays)

from pyrisk.game import Game
from AI.ppoagent import PPOAgent, PPOConfig
from AI.better_ai    import BetterAI
from AI.first_ai     import FirstAI
from AI.aggressive_ai import AggressiveAI
from AI.defensive_ai  import DefensiveAI
from AI.random_ai     import RandomAI

LOG = logging.getLogger("train")
AGENT_NAMES = ["PPO", "BetterAI", "FirstAI",
               "AggressiveAI", "DefensiveAI", "RandomAI"]

# ──────────────────────────────
#  Extra diagnostics (pickle+png)
# ──────────────────────────────
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

CRITIC_PKL  = RESULT_DIR / "critic_trace.pkl"
ACTOR_PKL   = Path("traces") / "strategy_probs_*.pkl"  # already emitted by PPOAgent
CRITIC_PNG  = RESULT_DIR / "critic_performance.png"
ACTOR_PNG   = RESULT_DIR / "actor_strategy_probs.png"

# ──────────────────────────────
#  CSV-logging (unchanged)
# ──────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

REWARD_CSV  = LOG_DIR / "rewards.csv"
COUNT_CSV   = LOG_DIR / "strategy_counts.csv"
SWITCH_CSV  = LOG_DIR / "strategy_switches.csv"

TRACK_FILE  = "ppo_training_progress.json"
MODEL_FILE  = "ppo_model_final.pt"

# initialise CSV files once
if not REWARD_CSV.exists():
    REWARD_CSV.write_text("episode,reward\n")
if not COUNT_CSV.exists():
    COUNT_CSV.write_text("episode,aggressive,defensive,balanced,random\n")
if not SWITCH_CSV.exists():
    SWITCH_CSV.write_text("episode,step,strategy\n")

# ──────────────────────────────
#  History helpers (unchanged)
# ──────────────────────────────
def load_history() -> List[Dict]:
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    return []

def save_history(history: List[Dict]) -> None:
    with open(TRACK_FILE, "w") as f:
        json.dump(history, f, indent=2)

def plot_history(history: List[Dict]) -> None:
    if not history:
        return
    games_cum, win_rates, total = [], [], 0
    for h in history:
        total += h.get("episodes", 0)
        games_cum.append(total)
        win_rates.append(h.get("win_rate", 0))
    plt.figure(figsize=(7, 4))
    plt.plot(games_cum, win_rates, marker="o")
    plt.title("PPO Win-rate Progress")
    plt.xlabel("Games"); plt.ylabel("Win-rate (%)"); plt.grid(True)
    plt.tight_layout()
    plt.savefig("ppo_winrate_plot_final3.png")
    LOG.info("Saved win-rate plot -> ppo_winrate_plot_final3.png")

# ──────────────────────────────
#  Game helpers
# ──────────────────────────────
def play_with_limits(game: Game, max_turns: int, factor: float = 0.06) -> str:
    """Run a game with a turn cap *and* wall-clock timeout."""
    timeout_s = max_turns * factor
    result: Dict[str, Optional[str]] = {"winner": None}
    def _runner():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sig = inspect.signature(game.play)
            result["winner"] = game.play(max_turns=max_turns) if "max_turns" in sig.parameters else game.play()
    th = threading.Thread(target=_runner, daemon=True)
    th.start(); th.join(timeout_s)
    if th.is_alive():                                   # timeout -> choose winner by troops
        LOG.warning("Timeout (%.1fs). Selecting leader by troop count.", timeout_s)
        alive = [p for p in game.players.values() if p.alive]
        return max(alive, key=lambda p: (p.forces, p.territory_count)).name if alive else "Draw"
    return result["winner"] or "Draw"

def make_ppo(player, game, world):
    return PPOAgent(player, game, world, config=PPOConfig(model_path=MODEL_FILE))

# ──────────────────────────────
#  Training loop
# ──────────────────────────────
def train(num_games: int, max_turns: int, log_int: int, seed: int, deal: bool) -> None:
    if seed != -1:
        random.seed(seed)

    wins = defaultdict(int, {n: 0 for n in AGENT_NAMES})
    history = load_history()
    previous_rate = history[-1]["win_rate"] if history else None

    # --- NEW lists for critic diagnostics ---
    critic_log:  list[float] = []
    return_log:  list[float] = []

    current_game: Optional[Game] = None
    try:
        for ep in range(1, num_games + 1):
            current_game = Game(curses=False, color=False, delay=0, wait=False, deal=deal)
            current_game.add_player("PPO", make_ppo)
            current_game.add_player("aggressive", AggressiveAI)
            winner = play_with_limits(current_game, max_turns)
            wins[winner] += 1

            # ── grab PPO episode summary ────────────────────────────────
            ppo_ai: PPOAgent = current_game.players["PPO"].ai  # type: ignore
            reward        = ppo_ai.episode_rewards[-1] if ppo_ai.episode_rewards else 0.0
            strat_counts  = ppo_ai.count
            switch_events = ppo_ai.switch_log

            # 1️⃣ reward CSV
            with REWARD_CSV.open("a") as f:
                f.write(f"{ep},{reward:.3f}\n")
            # 2️⃣ strategy count CSV
            with COUNT_CSV.open("a") as f:
                f.write(f"{ep},{strat_counts['aggressive']},{strat_counts['defensive']},"
                        f"{strat_counts['balanced']},{strat_counts['random']}\n")
            # 3️⃣ switch timing CSV
            if switch_events:
                with SWITCH_CSV.open("a") as f:
                    for step, strat in switch_events:
                        f.write(f"{ep},{step},{strat}\n")

            # --- NEW: collect critic traces for plotting later ----------
            #  (You must add `critic_trace` and `return_trace` lists in PPOAgent)
            critic_log.extend(ppo_ai.critic_trace)
            return_log.extend(ppo_ai.return_trace)

            # periodic console log
            if ep % log_int == 0 or ep == num_games:
                ppo_rate = wins["PPO"] / ep * 100
                delta    = ppo_rate - previous_rate if previous_rate is not None else 0.0
                LOG.info("[EP %d] PPO win-rate: %.2f%%  (Δ %.2f)", ep, ppo_rate, delta)

    except KeyboardInterrupt:
        LOG.warning("Interrupted!  Saving PPO model.")
        if current_game:
            current_game.players["PPO"].ai.end()        # flush + save
        return

    # ── summary & history tracking ──────────────────────────────────────
    total_rate = wins["PPO"] / num_games * 100
    history.append({"episodes": num_games, "win_rate": total_rate})
    save_history(history); plot_history(history)
    LOG.info("Finished %d games. Final PPO win-rate %.2f%%", num_games, total_rate)

    # ────────────────────────────────────────────────────────────────────
    #  NEW ▸ Dump critic trace + make both diagnostic plots
    # ────────────────────────────────────────────────────────────────────
    if critic_log and return_log:
        with CRITIC_PKL.open("wb") as f:
            import pickle
            pickle.dump({"pred": critic_log, "ret": return_log}, f)

        # critic plot
        plt.figure(figsize=(9, 4))
        plt.plot(return_log, label="Actual Returns")
        plt.plot(critic_log, label="Critic Predictions", alpha=.7)
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.title("Critic Predictions vs. Actual Returns")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(CRITIC_PNG)
        LOG.info("Saved critic plot -> %s", CRITIC_PNG)

    # actor plot (soft-max) – collected automatically by PPOAgent into traces/…
    import glob, pickle
    pkl_files = sorted(glob.glob(str(ACTOR_PKL)))
    if pkl_files:
        # merge all trace files into one array
        probs_all = []
        for path in pkl_files:
            with open(path, "rb") as f:
                while True:
                    try:
                        probs_all.extend(pickle.load(f))
                    except EOFError:
                        break
        if probs_all:
            probs_np = np.array([p[1] for p in probs_all])  # ignore step index
            labels = ["Aggressive", "Defensive", "Balanced", "Random"]
            plt.figure(figsize=(9, 4))
            for i in range(4):
                plt.plot(probs_np[:, i], label=labels[i])
            plt.xlabel("Segment")
            plt.ylabel("Probability")
            plt.title("Actor-Strategy Probabilities Over Time")
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(ACTOR_PNG)
            LOG.info("Saved actor probability plot -> %s", ACTOR_PNG)

# ──────────────────────────────
#  Entry-point
# ──────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PPO meta-agent with logging & diagnostics")
    p.add_argument("--episodes",     type=int, default=1)
    p.add_argument("--max_turns",    type=int, default=250)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--deal",         action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("pyrisk").setLevel(logging.ERROR)

    train(args.episodes, args.max_turns, args.log_interval, args.seed, args.deal)
