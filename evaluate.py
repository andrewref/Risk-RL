# evaluate.py
import logging
import random
from pyrisk.game import Game
from agents.ppoagent import PPOAgent
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("evaluate")

def evaluate(num_games=40, seed=42):
    random.seed(seed)
    wins = { 'PPO': 0, 'Aggressive': 0, 'Balanced': 0, 'Defensive': 0, 'Random': 0 }

    for episode in range(1, num_games + 1):
        g = Game(
            curses=False,
            color=False,
            delay=0,
            wait=False,
            deal=True  # optional: shuffle territories instead of manual claiming
        )

        g.add_player("PPO", PPOAgent)
        g.add_player("Aggressive", AggressiveAI)
        g.add_player("Balanced", BalancedAI)
        g.add_player("Defensive", DefensiveAI)
        g.add_player("Random", RandomAI)

        winner = g.play()
        wins[winner] = wins.get(winner, 0) + 1

        if episode % 10 == 0:
            LOG.info(f"Game {episode}/{num_games} done... Current Wins: {wins}")

    LOG.info("\n📊 Final Evaluation Results:")
    for k, v in wins.items():
        print(f"{k}: {v} wins")

if __name__ == "__main__":
    evaluate()
