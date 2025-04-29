"""
Evaluate your PPO agent vs the BetterAI scripted bot
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyrisk')))
import argparse
import numpy as np
from stable_baselines3 import PPO
from risk_env_multi import FourPlayerRiskEnv
from pyrisk.world import World
from pyrisk.ai.better import BetterAI
from pyrisk.player import Player  # ✨ Added this
from pyrisk.game import Game      # ✨ Added this

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PPO model .zip')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    return parser.parse_args()

def evaluate(model_path: str, episodes: int, seed: int):
    np.random.seed(seed)

    # load PPO model
    ppo = PPO.load(model_path, device='cuda')

    wins = {'ppo': 0, 'scripted': 0}

    for ep in range(episodes):
        env = FourPlayerRiskEnv()
        obs, _ = env.reset(seed=seed+ep)

        # Build fake Player, Game, World for BetterAI to think
        fake_world = env.world
        fake_player = Player("BetterBot", 1)
        fake_game = Game(["PPO", "BetterBot", "BetterBot", "BetterBot"])

        # Attach BetterAI
        scripted = BetterAI(fake_player, fake_game, fake_world)
        scripted.start()

        done = False
        while not done:
            pid = obs['pid']

            if pid == 0:
                # PPO plays
                action, _ = ppo.predict(obs, deterministic=True)
            else:
                # Scripted BetterAI plays
                legal_moves = env.action_list
                move_found = False

                for atk_from, atk_to_func, *_ in scripted.attack():
                    if (atk_from.name, atk_to_func.name) in legal_moves:
                        action = legal_moves.index((atk_from.name, atk_to_func.name))
                        move_found = True
                        break

                if not move_found:
                    action = len(legal_moves)  # noop

            obs, reward, done, trunc, _ = env.step(action)

        # Evaluate winner
        winner = env.world.current_player
        if winner == 0:
            wins['ppo'] += 1
        else:
            wins['scripted'] += 1

        print(f"[Episode {ep+1}/{episodes}] Winner → {'PPO' if winner == 0 else 'BetterAI'}")

    # Summary
    print("\n==== Final Results ====")
    print(f"PPO wins     : {wins['ppo']} / {episodes} ({wins['ppo']/episodes*100:.2f}%)")
    print(f"BetterAI wins: {wins['scripted']} / {episodes} ({wins['scripted']/episodes*100:.2f}%)")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.episodes, args.seed)
