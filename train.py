#!/usr/bin/env python3
import argparse
import logging
import random
from pyrisk.game import Game
from agents.ppoagent import PPOAgent
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

LOG = logging.getLogger("train")

def main(args):
    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)

    wins = {
        'PPO': 0,
        'AggressiveAI': 0,
        'BalancedAI': 0,
        'DefensiveAI': 0,
        'RandomAI': 0
    }

    try:
        for episode in range(1, args.episodes + 1):
            g = Game(
                curses=False,
                color=False,
                delay=0,
                wait=False,
                deal=args.deal
            )

            g.add_player("PPO", PPOAgent)
            g.add_player("AggressiveAI", AggressiveAI)
            g.add_player("BalancedAI", BalancedAI)
            g.add_player("DefensiveAI", DefensiveAI)
            g.add_player("RandomAI", RandomAI)

            winner = g.play()

            if winner in wins:
                wins[winner] += 1
            else:
                wins[winner] = 1

            if episode % args.log_interval == 0:
                LOG.info(f"Episode {episode}/{args.episodes} - Games won: {wins}")

    except KeyboardInterrupt:
        LOG.warning("⛔ Training interrupted by user.")
        try:
            # Force final PPO model save if current game is running
            for player_obj in g.players.values():
                if isinstance(player_obj.ai, PPOAgent):
                    player_obj.ai.end()  # Save model & update PPO one last time
                    LOG.info("✅ PPO model saved on interrupt.")
                    break
        except Exception as e:
            LOG.warning(f"⚠️ Could not save PPO model after interrupt: {e}")
        exit(0)

    LOG.info(f"✅ Training complete. Final win counts: {wins}")

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
