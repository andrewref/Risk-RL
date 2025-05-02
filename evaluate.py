"""
Evaluate your PPO agent vs the BetterAI scripted bot
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyrisk')))

import argparse
import numpy as np
from stable_baselines3 import PPO
from risk_env_multi import FourPlayerRiskEnv
from pyrisk.world import World
from pyrisk.ai.better import BetterAI
from pyrisk.player import Player
from pyrisk.game import Game

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, required=True,
                        help='Path to PPO model .zip')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--seed',     type=int, default=0,
                        help='Random seed')
    return parser.parse_args()

def evaluate(model_path: str, episodes: int, seed: int):
    np.random.seed(seed)
    ppo = PPO.load(model_path, device='cuda')
    wins = {'ppo': 0, 'scripted': 0}

    for ep in range(1, episodes+1):
        env = FourPlayerRiskEnv()
        obs, _ = env.reset(seed=seed+ep)

        # Set up BetterAI
        fake_world  = env.world
        fake_player = Player("BetterBot", 1)
        fake_game   = Game(["PPO", "BetterBot", "BetterBot", "BetterBot"])
        scripted    = BetterAI(fake_player, fake_game, fake_world)
        scripted.start()

        done = False
        while not done:
            pid = obs['pid']
            if pid == 0:
                action, _ = ppo.predict(obs, deterministic=True)
            else:
                legal_moves = env.action_list
                action = len(legal_moves)  # default noop
                for src, dst, *_ in scripted.attack():
                    key = (src.name, dst.name)
                    if key in legal_moves:
                        action = legal_moves.index(key)
                        break

            obs, _, done, _, _ = env.step(action)

        winner = env.world.current_player
        if winner == 0:
            wins['ppo'] += 1
        else:
            wins['scripted'] += 1

        print(f"[Episode {ep}/{episodes}] Winner → {'PPO' if winner==0 else 'BetterAI'}")

    print("\n==== Final Results ====")
    print(f"PPO wins     : {wins['ppo']} / {episodes} ({wins['ppo']/episodes*100:.2f}%)")
    print(f"BetterAI wins: {wins['scripted']} / {episodes} ({wins['scripted']/episodes*100:.2f}%)")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.episodes, args.seed)
