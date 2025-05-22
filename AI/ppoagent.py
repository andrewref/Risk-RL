import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from AI.aggressive_ai import AggressiveAI
from AI.balanced_ai import BalancedAI
from AI.defensive_ai import DefensiveAI
from AI.random_ai import RandomAI

LOG = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    seg_len: int = 3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    eps_start: float = 0.2
    eps_end: float = 0.01  # Increased from 0.0 to maintain some exploration
    eps_decay: float = 0.995
    k_epochs: int = 4
    entropy_coef: float = 0.02  # Increased for better exploration
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip_grad_norm: float = 0.5
    lr_schedule: bool = True
    total_updates: int = 5000
    model_path: str = "ppo_model_final.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ActorNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
        # Initialize with smaller values to prevent extreme policy at start
        nn.init.uniform_(self.net[-1].weight, -0.003, 0.003)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class CriticNet(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
        # Better initialization for critic
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.xavier_uniform_(self.net[4].weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)

class PPOAgent:
    def __init__(self, player, game, world, config: PPOConfig = PPOConfig()):
        self.player, self.game, self.world = player, game, world
        self.config = config
        self.device = torch.device(config.device)
        LOG.info(f"Using device: {self.device}")
        self.switch_log: list[tuple[int, str]] = []
        # Sub-strategies
        self.strategies: Dict[str, object] = {
            'aggressive': AggressiveAI(player, game, world),
            'balanced':  BalancedAI(player, game, world),
            'defensive': DefensiveAI(player, game, world),
            'random':    RandomAI(player, game, world)
        }
        self.names = list(self.strategies.keys())

        # Counter for how many times each strategy is chosen
        self.count = {n: 0 for n in self.names}

        # Networks & optimizers
        obs_dim = 8  # Increased state space for better representation
        self.actor   = ActorNet(obs_dim, len(self.names)).to(self.device)
        self.critic  = CriticNet(obs_dim).to(self.device)
        self.opt_actor  = optim.Adam(self.actor.parameters(), lr=config.lr_actor, eps=1e-5)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=config.lr_critic, eps=1e-5)

        # LR schedulers
        if config.lr_schedule:
            lr_lambda = lambda step: max(1 - step / config.total_updates, 0.1)  # Don't go to zero
            self.sched_actor  = LambdaLR(self.opt_actor, lr_lambda)
            self.sched_critic = LambdaLR(self.opt_critic, lr_lambda)

        # Memory buffers
        self.memory = {k: [] for k in ['state','action','logprob','reward','done','value']}

        # Runtime state
        self.step = 0
        self.current = 'random'
        self.prev_state: Tensor    = None
        self.prev_logprob: Tensor  = None
        self.prev_action: int      = 0
        self.prev_value: float     = 0
        self.snapshot_prev         = None
        self.episode_rewards = []

        # Exploration ε
        self.eps = config.eps_start

        # Load checkpoint if available
        self._load()

    def event(self, msg):
        for strat in self.strategies.values():
            if hasattr(strat, 'event'):
                strat.event(msg)

    def _load(self):
        if os.path.exists(self.config.model_path):
            try:
                data = torch.load(self.config.model_path, map_location=self.device)
                self.actor.load_state_dict(data.get('actor', {}))
                self.critic.load_state_dict(data.get('critic', {}))
                # Optional: Load optimizer state to continue training exactly
                if 'opt_actor' in data and 'opt_critic' in data:
                    self.opt_actor.load_state_dict(data['opt_actor'])
                    self.opt_critic.load_state_dict(data['opt_critic'])
                LOG.info(f"Loaded checkpoint {self.config.model_path}")
            except Exception as e:
                LOG.error(f"Failed to load checkpoint: {e}")

    def _save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'opt_actor': self.opt_actor.state_dict(),
            'opt_critic': self.opt_critic.state_dict(),
            'episode_rewards': self.episode_rewards
        }, self.config.model_path)
        LOG.info(f"Saved checkpoint to {self.config.model_path}")

    def _features(self) -> Tensor:
        """Enhanced feature extraction function"""
        p = self.game.players[self.player.name]
        # Get other players (enemies)
        enemies = [pl for name, pl in self.game.players.items() 
                  if name != self.player.name and pl.alive]
        
        # Calculate basic ratios
        t_all = sum(pl.territory_count for pl in self.game.players.values() if pl.alive)
        f_all = sum(pl.forces for pl in self.game.players.values() if pl.alive)
        
        # Enhanced territory features
        terr_ratio = p.territory_count / t_all if t_all else 0.0
        force_ratio = p.forces / f_all if f_all else 0.0
        area_ratio = sum(1 for _ in p.areas) / len(self.world.areas)
        
        # Border threat analysis
        enemy_forces, own_forces = [], []
        border_territories = 0
        total_borders = 0
        
        for terr in self.world.territories.values():
            if terr.owner == p:
                has_border = False
                for nb in terr.connect:
                    if nb.owner and nb.owner != p:
                        enemy_forces.append(nb.forces)
                        own_forces.append(terr.forces)
                        has_border = True
                        total_borders += 1
                if has_border:
                    border_territories += 1
        
        # Threat metrics
        threat = float(np.mean(enemy_forces)) if enemy_forces else 0.0
        defense_ratio = (float(np.mean(own_forces)) / max(float(np.mean(enemy_forces)), 1.0)) if enemy_forces else 1.0
        
        # Game progression feature (how far into the game we are)
        game_progress = min(self.step / 100.0, 1.0)  # Normalize by expected game length
        
        # Border vulnerability ratio
        border_ratio = border_territories / max(p.territory_count, 1)
        
        # Combine all features
        features = torch.tensor([
            terr_ratio, 
            force_ratio,
            area_ratio,
            threat,
            defense_ratio,
            game_progress,
            border_ratio,
            total_borders / max(p.territory_count, 1)
        ], device=self.device, dtype=torch.float32)
        
        return features

    def _snapshot(self) -> Dict[str, float]:
        """Enhanced game state snapshot with more metrics"""
        p = self.game.players[self.player.name]
        enemies = [pl for name, pl in self.game.players.items() 
                  if name != self.player.name and pl.alive]
        
        # Count border territories
        border_territories = 0
        for terr in self.world.territories.values():
            if terr.owner == p:
                for nb in terr.connect:
                    if nb.owner and nb.owner != p:
                        border_territories += 1
                        break
        
        return {
            'terr': p.territory_count,
            'forces': p.forces,
            'areas': sum(1 for _ in p.areas),
            'alive': int(p.alive),
            'border_terr': border_territories,
            'enemies': len(enemies),
            'enemy_terr': sum(e.territory_count for e in enemies) if enemies else 0,
            'enemy_forces': sum(e.forces for e in enemies) if enemies else 0,
        }

    def _gae(self, rewards: List[float], dones: List[int], values: List[float]) -> Tuple[Tensor, Tensor]:
        """More numerically stable GAE calculation"""
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[i + 1]
                
            next_non_terminal = 1.0 - dones[i]
            delta = rewards[i] + self.config.gamma * next_value * next_non_terminal - values[i]
            
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
            
        return (
            torch.tensor(advantages, device=self.device, dtype=torch.float32),
            torch.tensor(returns, device=self.device, dtype=torch.float32)
        )

    def _choose_action(self, state: Tensor) -> Tuple[int, Tensor, float]:
        """Choose action with epsilon-greedy strategy and return value estimate"""
        # Get value estimate
        with torch.no_grad():
            value = self.critic(state).item()
            
        # Epsilon-greedy action selection
        if np.random.rand() < self.eps:
            act = np.random.randint(len(self.names))
            logp = torch.log(torch.tensor(1.0/len(self.names), device=self.device))
        else:
            with torch.no_grad():
                logits = self.actor(state)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                act = dist.sample().item()
                logp = dist.log_prob(torch.tensor(act, device=self.device))
                
        return act, logp, value

    def _ppo_update(self) -> None:
        """Improved PPO update with value normalization and proper clipping"""
        if not self.memory['state']:  # Skip if no data
            return
            
        S = torch.stack(self.memory['state'])
        A = torch.tensor(self.memory['action'], device=self.device)
        old_logp = torch.stack(self.memory['logprob']).detach()
        R = self.memory['reward']
        D = self.memory['done']
        old_values = torch.tensor(self.memory['value'], device=self.device).detach()
        
        # Calculate advantages and returns
        adv, ret = self._gae(R, D, self.memory['value'])
        
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Mini-batch size - if data is small, use full batch
        batch_size = min(32, len(S))
        
        # Multiple epochs of training
        for _ in range(self.config.k_epochs):
            # Optional: Create mini-batches
            indices = np.random.permutation(len(S))
            
            for start_idx in range(0, len(S), batch_size):
                idx = indices[start_idx:start_idx + batch_size]
                
                # Get batch data
                b_states = S[idx]
                b_actions = A[idx]
                b_old_logp = old_logp[idx]
                b_adv = adv[idx]
                b_ret = ret[idx]
                b_old_v = old_values[idx]
                
                # Actor loss
                logits = self.actor(b_states)
                dist = Categorical(torch.softmax(logits, dim=-1))
                new_logp = dist.log_prob(b_actions)
                
                # Importance ratio and clipped objective
                ratio = torch.exp(new_logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1-self.config.eps_clip, 1+self.config.eps_clip) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                new_values = self.critic(b_states)
                critic_loss = 0.5 * ((new_values - b_ret) ** 2).mean()
                
                # Entropy bonus (to encourage exploration)
                entropy_loss = -self.config.entropy_coef * dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + critic_loss + entropy_loss
                
                # Update networks
                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                clip_grad_norm_(self.actor.parameters(), self.config.clip_grad_norm)
                clip_grad_norm_(self.critic.parameters(), self.config.clip_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()
                
        # Update learning rates
        if self.config.lr_schedule:
            self.sched_actor.step()
            self.sched_critic.step()
            
        # Decay exploration rate
        self.eps = max(self.eps * self.config.eps_decay, self.config.eps_end)
        
        # Log total episode reward
        if any(self.memory['done']):
            total_reward = sum(self.memory['reward'])
            self.episode_rewards.append(total_reward)
            LOG.info(f"Episode complete. Total reward: {total_reward:.2f}, Epsilon: {self.eps:.4f}")
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = sum(self.episode_rewards[-10:]) / 10
                LOG.info(f"Last 10 episodes avg reward: {avg_reward:.2f}")
        
        # Clear memory
        for k in self.memory:
            self.memory[k].clear()

    def start(self) -> None:
        """Initialize agent at the start of an episode"""
        self.step = 0
        self.current = 'random'  # Start with random strategy
        self.prev_state = None
        self.prev_logprob = None
        self.prev_action = self.names.index(self.current)
        self.prev_value = 0
        self.snapshot_prev = self._snapshot()
        self.switch_log.clear()
        # Reset counter each episode
        self.count = {n: 0 for n in self.names}
        self.trace_frames = []

        # Initialize sub-strategies
        for strat in self.strategies.values():
            strat.start()

    def initial_placement(self, *args, **kwargs):
        """Delegate initial placement to current strategy"""
        return self.strategies[self.current].initial_placement(*args, **kwargs)

    def attack(self):
        """Delegate attack to current strategy"""
        return self.strategies[self.current].attack()

    def freemove(self):
        """Delegate freemove to current strategy"""
        return self.strategies[self.current].freemove()

    def reinforce(self, troops: int):
        """Handle reinforcement phase and strategy selection"""
        if self.step and self.step % self.config.seg_len == 0:
            # Get current game state
            curr_snap = self._snapshot()
            
            # Calculate reward
            rew = self._compute_reward(self.snapshot_prev, curr_snap)
            
            # Store transition in memory
            if self.prev_state is not None:
                self.memory['state'].append(self.prev_state)
                self.memory['action'].append(self.prev_action)
                self.memory['logprob'].append(self.prev_logprob)
                self.memory['reward'].append(rew)
                self.memory['done'].append(False)
                self.memory['value'].append(self.prev_value)
            
            # Get new state and choose action
            state = self._features()
            act, logp, val = self._choose_action(state)
            
            # Update current state and action
            self.prev_state = state
            self.prev_action = act
            self.prev_logprob = logp
            self.prev_value = val
            self.current = self.names[act]
            self.switch_log.append((self.step, self.current))
            # Track strategy usage
            self.count[self.current] += 1
            
            # Update snapshot
            self.snapshot_prev = curr_snap
        if getattr(self, "trace_frames", None) is not None:
            self.trace_frames.append(self._snapshot())   # log a light-weight dict each turn    
        # Initialize state if this is the first step
        if self.prev_state is None:
            self.prev_state = self._features()
            act, logp, val = self._choose_action(self.prev_state)
            self.prev_action = act
            self.prev_logprob = logp
            self.prev_value = val
            self.current = self.names[act]
            self.switch_log.append((self.step, self.current))
            self.count[self.current] += 1
            
        self.step += 1
        return self.strategies[self.current].reinforce(troops)

    def end(self) -> None:
        """End of episode processing"""
        # Get final game state
        curr_snap = self._snapshot()
        
        # Calculate final reward
        rew = self._compute_reward(self.snapshot_prev, curr_snap)
        
        # Add final transition to memory
        if self.prev_state is not None:
            self.memory['state'].append(self.prev_state)
            self.memory['action'].append(self.prev_action)
            self.memory['logprob'].append(self.prev_logprob)
            self.memory['reward'].append(rew)
            self.memory['done'].append(True)
            self.memory['value'].append(self.prev_value)
        
        # Update policy if we have transitions
        if self.memory['state']:
            self._ppo_update()
            self._save()
        
        # End sub-strategies
        for strat in self.strategies.values():
            strat.end()
        if self.trace_frames and curr_snap['alive'] and curr_snap['enemies'] == 0:
          with open(f"traces/game_{self.player.game_id}.json", "w") as f:
                json.dump(self.trace_frames, f)
    
    def _compute_reward(self, prev: Dict[str, float], curr: Dict[str, float]) -> float:
        """Enhanced reward function that better captures progress and game dynamics"""
        # Territory and force changes
        terr_change = curr['terr'] - prev['terr']
        force_change = curr['forces'] - prev['forces']
        area_change = curr['areas'] - prev['areas']
        
        # Base reward
        r = (
            terr_change * 0.5 +             # Territory gain reward
            force_change * 0.1 +           # Force gain reward
            area_change * 2.0              # Area control reward
        )
        
        # Win/lose conditions with stronger penalties/rewards
        if not prev['alive'] and curr['alive']:  # Resurrection (shouldn't happen)
            r += 5.0
        if prev['alive'] and not curr['alive']:  # Death
            r -= 10.0
        
        # Enemy elimination bonus
        enemy_change = prev['enemies'] - curr['enemies']
        if enemy_change > 0:
            r += enemy_change * 3.0  # Big reward for eliminating enemies
            
        # Border territory control (strategic positioning)
        border_change = curr['border_terr'] - prev['border_terr']
        if terr_change > 0 and border_change < 0:
            # Bonus for consolidating territories (reducing border exposure)
            r += 0.5
            
        # Relative strength improvement
        if prev['enemy_forces'] > 0 and curr['enemy_forces'] > 0:
            prev_strength_ratio = prev['forces'] / prev['enemy_forces']
            curr_strength_ratio = curr['forces'] / curr['enemy_forces']
            
            if curr_strength_ratio > prev_strength_ratio:
                r += 0.3  # Reward for improving relative strength
        
        # Victory reward (if we're the only player alive)
        if curr['enemies'] == 0 and curr['alive']:
            r += 20.0  # Big reward for winning
            
        return float(r)  # Return as float (fixes the incomplete line in original code)
    def episode_summary(self) -> dict:
        return {
            "reward"   : self.episode_rewards[-1] if self.episode_rewards else 0.0,
            "counts"   : self.count.copy(),
            "switches" : self.switch_log.copy(),
        }