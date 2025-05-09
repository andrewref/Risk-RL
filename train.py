#!/usr/bin/env python3
import argparse
import logging
import random
from pyrisk.game import Game
from agents.ppoagent import PPOAgent  # ensure the filename matches your module
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

LOG = logging.getLogger("train") 
logging.basicConfig(level=logging.WARNING)#remove for terminal view 
import logging#remove for terminal view 
logging.getLogger("pyrisk.player").setLevel(logging.ERROR) #remove for terminal view 

def main(args):
    # Setup logging and random seed
    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)

    wins = { 'PPO': 0, 'AggressiveAI': 0, 'BalancedAI': 0, 'DefensiveAI': 0, 'RandomAI': 0 }

    for episode in range(1, args.episodes + 1):
        # Create a new game for each episode
        g = Game(
            curses=False,
            color=False,
            delay=0,
            wait=False,
            deal=args.deal
        )

        # Add players: PPO meta-agent and the four fixed strategies
        g.add_player("PPO", PPOAgent)
        g.add_player("Aggressive", AggressiveAI)
        g.add_player("Balanced", BalancedAI)
        g.add_player("Defensive", DefensiveAI)
        g.add_player("Random", RandomAI)

        # Play the game
        winner = g.play()

        # Record win
        wins[winner] = wins.get(winner, 0) + 1

        # Periodically log progress
        if episode % args.log_interval == 0:
            LOG.info(f"Episode {episode}/{args.episodes} - Wins so far: {wins}")

    # After training, final model was saved by PPOAgent.end()
    LOG.info(f"Training complete. Final win counts: {wins}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PPO meta-agent in PyRisk")
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Episodes between logging progress')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--deal', action='store_true', default=False,
                        help='Deal territories rather than choose')
    args = parser.parse_args()
    main(args)