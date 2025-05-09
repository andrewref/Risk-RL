import os
import time
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from pyrisk.game import Game
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI
from dynamic_strategy_agent import DynamicStrategyAgent

def train_agent(num_episodes=1000, save_interval=100, model_path="models/dynamic_strategy_agent",
                verbose=True, render=False, evaluation_interval=50):
    """
    Train the dynamic strategy agent.
    
    Args:
        num_episodes (int): Number of episodes to train for.
        save_interval (int): Interval for saving the model.
        model_path (str): Path to save/load the model.
        verbose (bool): Whether to print verbose output.
        render (bool): Whether to render the game.
        evaluation_interval (int): Interval for evaluating the agent.
    """
    # Create agent
    agent = DynamicStrategyAgent()
    
    # Try to load pre-trained model if exists
    try:
        agent.load_model(model_path)
        print("Loaded pre-trained model.")
    except:
        print("No pre-trained model found. Starting from scratch.")
    
    # Training metrics
    win_rates = []
    rewards = []
    
    # Training loop
    for episode in tqdm(range(1, num_episodes + 1)):
        # Create a new game for each episode
        game = Game()
        
        # Add players
        player_id = game.add_player(agent)
        
        # Add opponent AIs
        game.add_player(AggressiveAI)
        game.add_player(BalancedAI) 
        game.add_player(DefensiveAI)
        
        # Set game reference for agent
        agent.game = game
        agent.player_id = player_id
        
        # Initialize game
        game.initialize_game()
        
        # Play the game
        while not game.game_over:
            game.play_round()
            
            if render:
                os.system('clear' if os.name == 'posix' else 'cls')
                print(f"Episode: {episode}/{num_episodes}")
                print(f"Round: {game.round_num}")
                print(f"Current Strategy: {agent.current_strategy.name if agent.current_strategy else 'None'}")
                print(f"Rounds with current strategy: {agent.rounds_with_current_strategy}")
                game.print_world_state()
                time.sleep(0.1)
        
        # Update agent with game results
        agent.end_episode(won=(game.winner == player_id))
        
        # Save model periodically
        if episode % save_interval == 0:
            agent.save_model(model_path)
            
            if verbose:
                print(f"\nEpisode {episode}/{num_episodes}")
                agent.report_statistics()
        
        # Record metrics
        win_rates.append(agent.training_stats['wins'] / episode)
        if agent.training_stats['avg_rewards']:
            rewards.append(agent.training_stats['avg_rewards'][-1])
        
        # Evaluate agent periodically
        if episode % evaluation_interval == 0:
            eval_win_rate = evaluate_agent(agent, num_eval_episodes=10, verbose=False)
            print(f"\nEvaluation at episode {episode}: Win rate = {eval_win_rate:.2f}")
    
    # Final save
    agent.save_model(model_path)
    
    # Print final statistics
    if verbose:
        print("\nTraining completed!")
        agent.report_statistics()
    
    # Plot training metrics
    plot_training_metrics(win_rates, rewards)
    
    return agent

def evaluate_agent(agent, num_eval_episodes=100, verbose=True):
    """
    Evaluate the trained agent.
    
    Args:
        agent (DynamicStrategyAgent): The agent to evaluate.
        num_eval_episodes (int): Number of episodes to evaluate.
        verbose (bool): Whether to print verbose output.
    
    Returns:
        float: Win rate.
    """
    wins = 0
    
    for episode in tqdm(range(num_eval_episodes), desc="Evaluating", disable=not verbose):
        # Create a new game for each episode
        game = Game()
        
        # Add players
        player_id = game.add_player(agent)
        
        # Add opponent AIs
        game.add_player(AggressiveAI)
        game.add_player(BalancedAI)
        game.add_player(DefensiveAI)
        
        # Set game reference for agent
        agent.game = game
        agent.player_id = player_id
        
        # Initialize game
        game.initialize_game()
        
        # Play the game
        while not game.game_over:
            game.play_round()
        
        if game.winner == player_id:
            wins += 1
    
    win_rate = wins / num_eval_episodes
    
    if verbose:
        print(f"Evaluation results over {num_eval_episodes} episodes:")
        print(f"Win rate: {win_rate:.2f}")
    
    return win_rate

def play_game(agent_path="models/dynamic_strategy_agent", render=True):
    """
    Play a game with the trained agent.
    
    Args:
        agent_path (str): Path to load the agent model.
        render (bool): Whether to render the game.
    """
    # Create and load agent
    agent = DynamicStrategyAgent()
    try:
        agent.load_model(agent_path)
        print("Loaded trained agent.")
    except:
        print("No trained model found. Using untrained agent.")
    
    # Create game
    game = Game()
    
    # Add players
    player_id = game.add_player(agent)
    
    # Add opponent AIs
    game.add_player(AggressiveAI)
    game.add_player(BalancedAI)
    game.add_player(DefensiveAI)
    
    # Set game reference for agent
    agent.game = game
    agent.player_id = player_id
    
    # Initialize game
    game.initialize_game()
    
    # Play the game
    while not game.game_over:
        game.play_round()
        
        if render:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"Round: {game.round_num}")
            print(f"Current Strategy: {agent.current_strategy.name if agent.current_strategy else 'None'}")
            print(f"Rounds with current strategy: {agent.rounds_with_current_strategy}")
            game.print_world_state()
            time.sleep(0.5)
    
    # Print result
    if game.winner == player_id:
        print("You won!")
    else:
        print(f"Player {game.winner} won!")
    
    return game.winner == player_id

def plot_training_metrics(win_rates, rewards):
    """
    Plot training metrics.
    
    Args:
        win_rates (list): List of win rates.
        rewards (list): List of rewards.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot win rate
    plt.subplot(1, 2, 1)
    plt.plot(win_rates)
    plt.title("Win Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    
    # Plot rewards
    plt.subplot(1, 2, 2)
    plt.plot(rewards)
    plt.title("Average Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or play with the Risk dynamic strategy agent.')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'play'], default='train',
                        help='Mode to run in: train, evaluate, or play')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train for')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Interval for saving the model')
    parser.add_argument('--model-path', type=str, default="models/dynamic_strategy_agent",
                        help='Path to save/load the model')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--render', action='store_true',
                        help='Render the game')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Interval for evaluating the agent')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_path=args.model_path,
            verbose=args.verbose,
            render=args.render,
            evaluation_interval=args.eval_interval
        )
    elif args.mode == 'evaluate':
        agent = DynamicStrategyAgent()
        try:
            agent.load_model(args.model_path)
            print(f"Loaded model from {args.model_path}")
        except:
            print("No trained model found. Using untrained agent.")
        
        win_rate = evaluate_agent(agent, num_eval_episodes=args.eval_episodes, verbose=args.verbose)
        print(f"Evaluation win rate: {win_rate:.2f}")
    elif args.mode == 'play':
        play_game(agent_path=args.model_path, render=True)