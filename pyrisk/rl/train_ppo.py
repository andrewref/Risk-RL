import argparse
import random
from datetime import datetime
from pathlib import Path

from pyrisk.game import Game
from AI.ppoagent import PPOAgent, PPOConfig
from AI.aggressive_ai import AggressiveAI
from AI.defensive_ai import DefensiveAI
from AI.balanced_ai import BalancedAI
from AI.random_ai import RandomAI

# Opponent distribution: two powerful (Aggressive, Defensive),
# one normal (Balanced), one easy (Random)
OPPONENTS = [
    ("AggressiveAI", AggressiveAI, 0.3),
    ("DefensiveAI", DefensiveAI, 0.3),
    ("BalancedAI", BalancedAI, 0.2),
    ("RandomAI", RandomAI, 0.2),
]


def choose_opponent() -> tuple[str, type]:
    r = random.random()
    cum = 0.0
    for name, cls, prob in OPPONENTS:
        cum += prob
        if r <= cum:
            return name, cls
    return OPPONENTS[-1][:2]


def main(episodes: int, model_path: str, fresh: bool) -> None:
    """Train PPO agent from scratch against a mix of scripted opponents."""
    if fresh:
        Path(model_path).unlink(missing_ok=True)
    config = PPOConfig(model_path=model_path)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    for ep in range(episodes):
        opp_name, opp_class = choose_opponent()
        game = Game(curses=False, color=False, delay=0.0, run_id=run_id,
                    round=(ep + 1, episodes))
        game.add_player("PPO", PPOAgent, use_trained=True, config=config)
        game.add_player(opp_name, opp_class)
        winner = game.play()
        print(f"Episode {ep + 1}/{episodes} vs {opp_name}: winner {winner}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent from scratch")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of games to play for training")
    parser.add_argument("--model-path", type=str, default="ppo_model.pt",
                        help="Where to save/load the PPO weights")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing model file to start from scratch")
    args = parser.parse_args()
    main(args.episodes, args.model_path, args.fresh)
