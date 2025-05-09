# ppoagent.py
import os
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

LOG = logging.getLogger("pyrisk.player.PPOAgent")

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=64, lr=1e-3):
        super(ActorNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.layers(x)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, lr=1e-3):
        super(CriticNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.layers(x)

class PPOAgent:
    def __init__(self, player, game, world, **kwargs):
        # Core game components
        self.player = player
        self.game = game
        self.world = world
        self.player_name = player.name

        # Strategy pool
        self.strategies = {
            'aggressive': AggressiveAI(self.player, self.game, self.world),
            'balanced':   BalancedAI(self.player, self.game, self.world),
            'defensive':  DefensiveAI(self.player, self.game, self.world),
            'random':     RandomAI(self.player, self.game, self.world)
        }
        self.strategy_list = list(self.strategies.keys())
        self.current_strategy = 'balanced'

        # PPO settings
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.segment_len = 7

        # Feature & action sizes
        self.input_dim = 5  # [territory_ratio, force_ratio, control, threat, weakness]
        self.n_actions = len(self.strategy_list)

        # Networks
        self.actor = ActorNetwork(self.input_dim, self.n_actions)
        self.critic = CriticNetwork(self.input_dim)

        # Memory buffers
        self.memory = { 'states': [], 'actions': [], 'log_probs': [],
                        'rewards': [], 'dones': [] }
        self.step_count = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None

        # Persistence
        self.model_path = 'ppo_model.pt'
        if os.path.exists(self.model_path):
            self.load_model()

    def extract_features(self):
        p = self.game.players[self.player_name]
        # Territory & force ratios
        total_terr = sum(pl.territory_count for pl in self.game.players.values() if pl.alive)
        total_forc = sum(pl.forces for pl in self.game.players.values() if pl.alive)
        t_ratio = p.territory_count / total_terr if total_terr > 0 else 0.0
        f_ratio = p.forces / total_forc if total_forc > 0 else 0.0

        # Control: how many areas we fully own
        control = sum(1 for _ in p.areas) / len(self.world.areas)

        # Threat & weakness around borders
        enemy, own = [], []
        for terr in self.world.territories.values():
            if terr.owner == p:
                for nb in terr.connect:
                    if nb.owner and nb.owner != p:
                        enemy.append(nb.forces)
                        own.append(terr.forces)
        threat = float(np.mean(enemy)) if enemy else 0.0
        weakness = float(np.mean(own) / np.mean(enemy)) if (own and enemy and np.mean(enemy) > 0) else 0.0

        return torch.tensor([t_ratio, f_ratio, control, threat, weakness], dtype=torch.float32)

    def select_strategy(self, state):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_reward(self, prev, curr):
        r = (curr[0] - prev[0]) + (curr[1] - prev[1])
        alive_bonus = 1.0 if self.game.players[self.player_name].alive else -1.0
        return (r + 0.1 * alive_bonus).item()

    def update_policy(self):
        # Stack memory
        states = torch.stack(self.memory['states'])
        actions = torch.tensor(self.memory['actions'], dtype=torch.long)
        old_log = torch.stack(self.memory['log_probs']).detach()
        rewards = self.memory['rewards']
        dones = self.memory['dones']

        # Compute returns
        returns = []
        G = 0.0
        for rew, done in zip(reversed(rewards), reversed(dones)):
            G = rew + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Advantages
        values = self.critic(states).squeeze()
        advs = returns - values.detach()

        # PPO optimization
        for _ in range(self.k_epochs):
            new_probs = self.actor(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log = dist.log_prob(actions)
            new_vals = self.critic(states).squeeze()

            ratios = torch.exp(new_log - old_log)
            s1 = ratios * advs
            s2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advs

            lossA = -torch.min(s1, s2).mean()
            lossC = F.mse_loss(new_vals.view(-1), returns.view(-1)) 
            loss = lossA + 0.5 * lossC

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        # Clear memory
        for k in self.memory:
            self.memory[k].clear()

    def save_model(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, self.model_path)
        LOG.info(f"Saved PPO model to {self.model_path}")

    def load_model(self):
        data = torch.load(self.model_path)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        LOG.info(f"Loaded PPO model from {self.model_path}")

    # PyRisk callbacks
    def event(self, msg):
        for strat in self.strategies.values():
            if hasattr(strat, 'event'):
                strat.event(msg)

    def start(self):
        self.step_count = 0
        self.prev_state = None
        for strat in self.strategies.values():
            strat.start()

    def initial_placement(self, empty, remaining):
        return self.strategies[self.current_strategy].initial_placement(empty, remaining)

    def reinforce(self, reinforcements):
        # Evaluate and record segments
        if self.step_count % self.segment_len == 0:
            st = self.extract_features()
            if self.prev_state is not None:
                r = self.compute_reward(self.prev_state, st)
                self.memory['states'].append(self.prev_state)
                self.memory['actions'].append(self.prev_action)
                self.memory['log_probs'].append(self.prev_log_prob)
                self.memory['rewards'].append(r)
                self.memory['dones'].append(0)
            a, lp = self.select_strategy(st)
            self.current_strategy = self.strategy_list[a]
            self.prev_state, self.prev_action, self.prev_log_prob = st, a, lp
        self.step_count += 1
        return self.strategies[self.current_strategy].reinforce(reinforcements)

    def attack(self):
        return self.strategies[self.current_strategy].attack()

    def freemove(self):
        return self.strategies[self.current_strategy].freemove()

    def end(self):
        # Final segment
        st = self.extract_features()
        if self.prev_state is not None:
            r = self.compute_reward(self.prev_state, st)
            self.memory['states'].append(self.prev_state)
            self.memory['actions'].append(self.prev_action)
            self.memory['log_probs'].append(self.prev_log_prob)
            self.memory['rewards'].append(r)
            self.memory['dones'].append(1)
        # Update and persist
        self.update_policy()
        self.save_model()
        for strat in self.strategies.values():
            strat.end()