from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

# Strategy constants
STRATEGIES = [AggressiveAI, BalancedAI, DefensiveAI]
STRATEGY_NAMES = ["Aggressive", "Balanced", "Defensive"]
NUM_STRATEGIES = len(STRATEGIES)
STRATEGY_SWITCH_INTERVAL = 7  # Switch strategy consideration every 7 rounds

# PPO Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
BATCH_SIZE = 64
MEMORY_SIZE = 1000  # Size of replay buffer

class DynamicStrategyPPONetwork(nn.Module):
    """
    Neural network for the PPO agent that decides which strategy to use.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(DynamicStrategyPPONetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns policy logits and value estimate for the given state.
        """
        features = self.feature_extractor(state)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select an action based on the current policy.
        Returns action index, log probability, and value estimate.
        """
        policy_logits, value = self.forward(state)
        
        # Create a distribution over actions
        dist = Categorical(logits=policy_logits)
        
        # Sample action or take most likely action if deterministic
        if deterministic:
            action = torch.argmax(policy_logits).item()
        else:
            action = dist.sample().item()
        
        # Get log probability of the selected action
        log_prob = dist.log_prob(torch.tensor(action))
        
        return action, log_prob.item(), value.item()

class PPOMemory:
    """Memory buffer for PPO training."""
    def __init__(self, batch_size=64):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.actions), \
               np.array(self.log_probs), np.array(self.rewards), \
               np.array(self.values), np.array(self.dones), batches

class DynamicStrategyAgent:
    """
    Agent that dynamically switches between different strategies based on game state.
    Uses PPO to learn when to switch strategies.
    """
    def __init__(self, game=None, player_id=None):
        self.game = game
        self.player_id = player_id
        self.name = "DynamicStrategyAgent"
        
        # State and action dimensions
        self.state_dim = self._get_state_dim()
        self.action_dim = NUM_STRATEGIES
        
        # Initialize PPO network and optimizer
        self.policy = DynamicStrategyPPONetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        
        # Initialize memory buffer
        self.memory = PPOMemory(BATCH_SIZE)
        
        # Strategy tracking
        self.current_strategy = None
        self.current_strategy_idx = None
        self.rounds_with_current_strategy = 0
        self.performance_history = []
        
        # Game state tracking
        self.prev_territories_count = 0
        self.prev_armies_count = 0
        self.episode_rewards = []
        self.cumulative_reward = 0
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=MEMORY_SIZE)
        
        # Training metrics
        self.training_stats = {
            'episodes': 0,
            'wins': 0,
            'losses': 0,
            'avg_rewards': [],
            'strategy_usage': {name: 0 for name in STRATEGY_NAMES}
        }
    
    def _get_state_dim(self) -> int:
        """
        Determine the state dimension for the policy network.
        This includes features like:
        - Territory control percentages
        - Army counts
        - Continent control
        - Threat levels
        - Current round
        """
        # Basic state features
        return 15  # We'll use 15 state features
    
    def _extract_state_features(self) -> np.ndarray:
        """
        Extract relevant state features from the game state.
        """
        if not self.game:
            # Return a default state if game is not set
            return np.zeros(self.state_dim)
        
        # Get game state information
        players = self.game.players
        world = self.game.world
        player = players[self.player_id]
        
        # Calculate basic features
        total_territories = len(world.territories)
        total_armies = sum(t.armies for t in world.territories.values())
        
        # Player-specific features
        player_territories = len(player.territories)
        player_armies = sum(t.armies for t in world.territories.values() if t.owner == self.player_id)
        
        # Territory control percentage
        territory_control = player_territories / total_territories if total_territories > 0 else 0
        
        # Army strength percentage
        army_strength = player_armies / total_armies if total_armies > 0 else 0
        
        # Continent control (how many continents the player fully controls)
        continent_control = sum(1 for c in world.continents.values() 
                               if all(t in player.territories for t in c.territories))
        continent_control_ratio = continent_control / len(world.continents)
        
        # Opponent information
        opponents = [p for pid, p in players.items() if pid != self.player_id]
        strongest_opponent_armies = max([sum(t.armies for t in world.territories.values() if t.owner == p.player_id) 
                                      for p in opponents]) if opponents else 0
        strongest_opponent_territories = max([len(p.territories) for p in opponents]) if opponents else 0
        
        # Relative strength
        relative_army_strength = player_armies / (strongest_opponent_armies + 1)  # +1 to avoid division by zero
        relative_territory_strength = player_territories / (strongest_opponent_territories + 1)
        
        # Border vulnerability (ratio of player territories adjacent to enemy territories)
        border_territories = sum(1 for t in player.territories 
                                if any(adj not in player.territories for adj in world.territories[t].adjacent))
        border_vulnerability = border_territories / player_territories if player_territories > 0 else 1
        
        # Game progression
        current_round = self.game.round_num / 50  # Normalize assuming max 50 rounds
        
        # Performance metrics from previous rounds
        territory_change = (player_territories - self.prev_territories_count) / max(1, self.prev_territories_count) if self.prev_territories_count > 0 else 0
        army_change = (player_armies - self.prev_armies_count) / max(1, self.prev_armies_count) if self.prev_armies_count > 0 else 0
        
        # Update previous counts
        self.prev_territories_count = player_territories
        self.prev_armies_count = player_armies
        
        # Create feature vector
        features = np.array([
            territory_control,
            army_strength,
            continent_control_ratio,
            relative_army_strength,
            relative_territory_strength,
            border_vulnerability,
            current_round,
            territory_change,
            army_change,
            player_territories / total_territories,
            player_armies / total_armies,
            strongest_opponent_territories / total_territories,
            strongest_opponent_armies / total_armies,
            self.current_strategy_idx / NUM_STRATEGIES if self.current_strategy_idx is not None else 0,
            self.rounds_with_current_strategy / STRATEGY_SWITCH_INTERVAL
        ])
        
        return features
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on game state changes.
        Higher reward for territory gains, army gains, and continent control.
        """
        if not self.game:
            return 0.0
        
        player = self.game.players[self.player_id]
        world = self.game.world
        
        # Basic metrics
        player_territories = len(player.territories)
        player_armies = sum(t.armies for t in world.territories.values() if t.owner == self.player_id)
        
        # Territory and army changes
        territory_change = player_territories - self.prev_territories_count
        army_change = player_armies - self.prev_armies_count
        
        # Continent control
        continent_control = sum(1 for c in world.continents.values() 
                               if all(t in player.territories for t in c.territories))
        
        # Calculate reward components
        territory_reward = territory_change * 2.0  # Territory gains are important
        army_reward = army_change * 0.1  # Army gains are moderately important
        continent_reward = continent_control * 5.0  # Continent control is very important
        
        # Add survival reward
        survival_reward = 0.1  # Small reward for surviving each round
        
        # Calculate total reward
        reward = territory_reward + army_reward + continent_reward + survival_reward
        
        # Add win/loss reward
        if self.game.winner == self.player_id:
            reward += 100.0  # Big reward for winning
        elif self.game.winner is not None and self.game.winner != self.player_id:
            reward -= 50.0  # Penalty for losing
        
        return reward
    
    def _select_strategy(self, state_features: np.ndarray, evaluate: bool = False) -> int:
        """
        Select a strategy using the PPO policy.
        """
        state_tensor = torch.FloatTensor(state_features)
        
        # Get action from policy
        action, log_prob, value = self.policy.get_action(state_tensor, deterministic=evaluate)
        
        return action, log_prob, value
    
    def update_strategy(self, evaluate: bool = False) -> None:
        """
        Update the current strategy based on the policy.
        """
        # Extract state features
        state_features = self._extract_state_features()
        
        # Check if it's time to switch strategies
        if self.rounds_with_current_strategy >= STRATEGY_SWITCH_INTERVAL or self.current_strategy is None:
            # Select a new strategy
            strategy_idx, log_prob, value = self._select_strategy(state_features, evaluate)
            
            # Create new strategy instance
            strategy_class = STRATEGIES[strategy_idx]
            self.current_strategy = strategy_class(self.game, self.player_id)
            self.current_strategy_idx = strategy_idx
            
            # Reset rounds counter
            self.rounds_with_current_strategy = 0
            
            # Calculate reward for the previous strategy
            reward = self._calculate_reward()
            
            # Store experience in memory
            if not evaluate:
                self.memory.store(
                    state_features,
                    strategy_idx,
                    log_prob,
                    reward,
                    value,
                    False  # done flag
                )
                
                # Update cumulative reward
                self.cumulative_reward += reward
                self.episode_rewards.append(reward)
                
                # Update strategy usage statistics
                self.training_stats['strategy_usage'][STRATEGY_NAMES[strategy_idx]] += 1
        else:
            # Continue with current strategy
            self.rounds_with_current_strategy += 1
    
    def train_ppo(self) -> dict:
        """
        Train the PPO network using collected experiences.
        """
        if len(self.memory.states) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        
        # Generate batches from memory
        states, actions, old_log_probs, rewards, old_values, dones, batches = self.memory.generate_batches()
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Calculate advantages
        advantages = np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount * (rewards[k] + GAMMA * old_values[k+1] * (1-dones[k]) - old_values[k])
                discount *= GAMMA * GAE_LAMBDA
            advantages[t] = a_t
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute value targets
        value_targets = advantages + torch.FloatTensor(old_values)
        
        # PPO update for multiple epochs
        for _ in range(PPO_EPOCHS):
            for batch in batches:
                # Get states, actions, etc. for this batch
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch]
                batch_value_targets = value_targets[batch]
                
                # Forward pass
                policy_logits, values = self.policy(batch_states)
                
                # Create distribution
                dist = Categorical(logits=policy_logits)
                
                # Calculate new log probs
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-PPO_CLIP, 1+PPO_CLIP) * batch_advantages
                
                # Calculate actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate critic loss
                critic_loss = nn.MSELoss()(values.squeeze(), batch_value_targets)
                
                # Calculate entropy
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + VALUE_LOSS_COEF * critic_loss - ENTROPY_COEF * entropy
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
        
        # Clear memory after update
        self.memory.clear()
        
        return {
            "policy_loss": actor_loss.item(),
            "value_loss": critic_loss.item(),
            "entropy": entropy.item()
        }
    
    def end_episode(self, won: bool) -> None:
        """
        Handle end of episode, update statistics, and train.
        """
        # Update win/loss statistics
        self.training_stats['episodes'] += 1
        if won:
            self.training_stats['wins'] += 1
        else:
            self.training_stats['losses'] += 1
        
        # Store average reward
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.training_stats['avg_rewards'].append(avg_reward)
        
        # Train policy
        training_stats = self.train_ppo()
        
        # Reset episode variables
        self.episode_rewards = []
        self.cumulative_reward = 0
        self.current_strategy = None
        self.current_strategy_idx = None
        self.rounds_with_current_strategy = 0
        self.prev_territories_count = 0
        self.prev_armies_count = 0
    
    def save_model(self, path: str = "models/dynamic_strategy_agent") -> None:
        """
        Save model weights and training statistics.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model weights
        torch.save(self.policy.state_dict(), f"{path}_policy.pt")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), f"{path}_optimizer.pt")
        
        # Save training statistics
        with open(f"{path}_stats.pkl", "wb") as f:
            pickle.dump(self.training_stats, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = "models/dynamic_strategy_agent") -> None:
        """
        Load model weights and training statistics.
        """
        # Load model weights
        self.policy.load_state_dict(torch.load(f"{path}_policy.pt"))
        
        # Load optimizer state
        self.optimizer.load_state_dict(torch.load(f"{path}_optimizer.pt"))
        
        # Load training statistics
        try:
            with open(f"{path}_stats.pkl", "rb") as f:
                self.training_stats = pickle.load(f)
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print("Training statistics file not found, using default values")
    
    def report_statistics(self) -> None:
        """
        Report training statistics.
        """
        print(f"Episodes: {self.training_stats['episodes']}")
        print(f"Wins: {self.training_stats['wins']}")
        print(f"Win rate: {self.training_stats['wins'] / max(1, self.training_stats['episodes']):.2f}")
        
        if self.training_stats['avg_rewards']:
            print(f"Average reward: {sum(self.training_stats['avg_rewards']) / len(self.training_stats['avg_rewards']):.2f}")
        
        print("Strategy usage:")
        for strategy, count in self.training_stats['strategy_usage'].items():
            print(f"  {strategy}: {count} ({count / max(1, self.training_stats['episodes']):.2f})")
    
    # Game interface methods
    def reinforce(self) -> None:
        """Interface method for game - reinforcement phase."""
        # Update strategy if needed
        self.update_strategy()
        
        # Use the current strategy's method
        self.current_strategy.reinforce()
    
    def attack(self) -> bool:
        """Interface method for game - attack phase."""
        # Update strategy if needed
        self.update_strategy()
        
        # Use the current strategy's method
        return self.current_strategy.attack()
    
    def fortify(self) -> None:
        """Interface method for game - fortify phase."""
        # Update strategy if needed
        self.update_strategy()
        
        # Use the current strategy's method
        self.current_strategy.fortify()