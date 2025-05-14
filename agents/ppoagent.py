import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai import BalancedAI
from agents.defensive_ai import DefensiveAI
from agents.random_ai import RandomAI

LOG = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    seg_len: int = 7                     # steps per update segment
    gamma: float = 0.99                  # discount factor
    gae_lambda: float = 0.95             # GAE lambda
    eps_clip: float = 0.2                # PPO clip threshold
    k_epochs: int = 4                    # PPO update epochs
    entropy_coef: float = 0.01           # entropy regularization weight
    reward_scale: float = 10.0           # reward scaling factor
    lr_actor: float = 3e-4               # actor learning rate
    lr_critic: float = 1e-3              # critic learning rate
    clip_grad_norm: float = 0.5          # gradient norm clipping
    lr_schedule: bool = True             # enable LR annealing
    total_updates: int = 10000           # total updates for annealing schedule
    model_path: str = "ppo_model.pt"    # checkpoint path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ActorNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
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

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)

class PPOAgent:
    """
    PPO meta-agent selecting among sub-strategies with GAE, LR scheduling,
    gradient clipping, and event propagation.
    """

    def __init__(self, player, game, world, config: PPOConfig = PPOConfig()):
        self.player, self.game, self.world = player, game, world
        self.config = config
        self.device = torch.device(config.device)
        LOG.info(f"Using device: {self.device}")

        # Sub-strategies
        self.strategies: Dict[str, object] = {
            'aggressive': AggressiveAI(player, game, world),
            'balanced': BalancedAI(player, game, world),
            'defensive': DefensiveAI(player, game, world),
            'random': RandomAI(player, game, world)
        }
        self.names = list(self.strategies.keys())
        obs_dim = 5

        # Networks
        self.actor = ActorNet(obs_dim, len(self.names)).to(self.device)
        self.critic = CriticNet(obs_dim).to(self.device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        # LR schedulers
        if config.lr_schedule:
            lr_lambda = lambda step: max(1 - step / config.total_updates, 0)
            self.sched_actor = LambdaLR(self.opt_actor, lr_lambda)
            self.sched_critic = LambdaLR(self.opt_critic, lr_lambda)

        # Memory buffers
        self.memory = {k: [] for k in ['state','action','logprob','reward','done']}

        # Runtime state
        self.step = 0
        self.current = 'random'
        self.prev_state: Tensor = None
        self.prev_logprob: Tensor = None
        self.prev_action: int = 0
        self.snapshot_prev = None

        # Exploration & biases
        self.eps = config.eps_clip
        self.bias = {n:1.0 for n in self.names}
        self.punish = {n:0 for n in self.names}
        self.usage = {n:0 for n in self.names}
        self.count = {n:0 for n in self.names}
        self.returns = {n:[] for n in self.names}

        # Load checkpoint
        self._load()

    def event(self, msg):
        """Propagate game events to sub-strategies."""
        for strat in self.strategies.values():
            if hasattr(strat, 'event'):
                strat.event(msg)

    def _load(self):
        if os.path.exists(self.config.model_path):
            data = torch.load(self.config.model_path, map_location=self.device)
            self.actor.load_state_dict(data.get('actor', {}), strict=False)
            self.critic.load_state_dict(data.get('critic', {}), strict=False)
            LOG.info(f"Loaded checkpoint {self.config.model_path} (strict=False)")

    def _save(self):
        torch.save({'actor':self.actor.state_dict(), 'critic':self.critic.state_dict()}, self.config.model_path)
        LOG.info(f"Saved checkpoint to {self.config.model_path}")

    def _features(self) -> Tensor:
        p = self.game.players[self.player.name]
        t_all = sum(pl.territory_count for pl in self.game.players.values() if pl.alive)
        f_all = sum(pl.forces for pl in self.game.players.values() if pl.alive)
        ratios = torch.tensor([
            p.territory_count/t_all if t_all else 0.0,
            p.forces/f_all if f_all else 0.0,
            sum(1 for _ in p.areas)/len(self.world.areas)
        ], device=self.device)
        enemy, own = [], []
        for terr in self.world.territories.values():
            if terr.owner == p:
                for nb in terr.connect:
                    if nb.owner and nb.owner != p:
                        enemy.append(nb.forces)
                        own.append(terr.forces)
        threat = float(np.mean(enemy)) if enemy else 0.0
        strength = float(np.mean(own)/np.mean(enemy)) if enemy and np.mean(enemy)>0 else 0.0
        return torch.cat([ratios, torch.tensor([threat, strength], device=self.device)])

    def _snapshot(self) -> Dict[str, float]:
        p = self.game.players[self.player.name]
        return {
            'terr': p.territory_count,
            'forces': p.forces,
            'areas': sum(1 for _ in p.areas),
            'alive': int(p.alive)
        }

    def _gae(self, rewards: List[float], dones: List[int], values: Tensor) -> Tuple[Tensor, Tensor]:
        gae = 0
        advantages = []
        vals = values.detach().cpu().numpy().tolist() + [0]
        for r, d, v, v_next in zip(reversed(rewards), reversed(dones), reversed(vals[:-1]), reversed(vals[1:])):
            delta = r + self.config.gamma * v_next * (1-d) - v
            gae = delta + self.config.gamma * self.config.gae_lambda * (1-d) * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, vals[:-1])]
        return torch.tensor(advantages, device=self.device), torch.tensor(returns, device=self.device)

    def _choose_action(self, state: Tensor) -> Tuple[int, Tensor]:
        if np.random.rand() < self.eps:
            act = np.random.randint(len(self.names))
            logp = torch.log(torch.tensor(1/len(self.names), device=self.device))
        else:
            logits = self.actor(state)
            probs = torch.softmax(logits, dim=-1)
            bias_arr = torch.tensor([self.bias[n] for n in self.names], device=self.device)
            probs = (probs * bias_arr).clamp(min=1e-8)
            probs /= probs.sum()
            dist = Categorical(probs)
            act = dist.sample().item()
            logp = dist.log_prob(torch.tensor(act, device=self.device))
        return act, logp

    def _ppo_update(self) -> None:
        S = torch.stack(self.memory['state'])
        A = torch.tensor(self.memory['action'], device=self.device)
        old_logp = torch.stack(self.memory['logprob']).detach()
        R, D = self.memory['reward'], self.memory['done']
        with torch.no_grad(): vals = self.critic(S)
        adv, ret = self._gae(R, D, vals)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(self.config.k_epochs):
            dist = Categorical(torch.softmax(self.actor(S), dim=-1))
            new_logp = dist.log_prob(A)
            ratio = (new_logp - old_logp).exp()
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1-self.config.eps_clip, 1+self.config.eps_clip) * adv)
            loss = -surr.mean() + 0.5 * nn.MSELoss()(self.critic(S), ret) - self.config.entropy_coef * dist.entropy().mean()
            self.opt_actor.zero_grad(); self.opt_critic.zero_grad()
            loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.config.clip_grad_norm)
            clip_grad_norm_(self.critic.parameters(), self.config.clip_grad_norm)
            self.opt_actor.step(); self.opt_critic.step()
        if self.config.lr_schedule:
            self.sched_actor.step(); self.sched_critic.step()
        for k in self.memory: self.memory[k].clear()

    def start(self) -> None:
        self.step = 0; self.current = 'random'; self.prev_state = None; self.prev_logprob = None
        self.prev_action = self.names.index(self.current)
        self.snapshot_prev = self._snapshot(); self.eps = self.config.eps_clip
        for n in self.names: self.bias[n] = 1.0; self.punish[n] = 0; self.usage[n] = 0; self.count[n] = 0; self.returns[n] = []
        for strat in self.strategies.values(): strat.start()

    def initial_placement(self, *args, **kwargs):
        return self.strategies[self.current].initial_placement(*args, **kwargs)

    def attack(self):
        return self.strategies[self.current].attack()

    def freemove(self):
        return self.strategies[self.current].freemove()

    def reinforce(self, troops: int) -> List:
        if self.step and self.step % self.config.seg_len == 0:
            curr_snap = self._snapshot()
            rew = self._compute_reward(self.snapshot_prev, curr_snap)
            self.memory['state'].append(self.prev_state)
            self.memory['action'].append(self.prev_action)
            self.memory['logprob'].append(self.prev_logprob)
            self.memory['reward'].append(rew)
            self.memory['done'].append(False)
            self.returns[self.current].append(rew)
            if rew < 0:
                self.punish[self.current] += 1
                self.bias[self.current] = max(0.2, self.bias[self.current] - 0.1)
            else:
                self.bias[self.current] = min(3.0, self.bias[self.current] + 0.1)
            state, act = self._features(), None
            act, logp = self._choose_action(state)
            self.prev_state, self.prev_action, self.prev_logprob = state, act, logp
            self.current = self.names[act]
            self.usage[self.current] += 1
            self.count[self.current] += 1
            self.snapshot_prev = curr_snap
        if self.prev_state is None:
            self.prev_state = self._features()
            self.prev_logprob = torch.log(torch.tensor(1/len(self.names), device=self.device))
        self.step += 1
        return self.strategies[self.current].reinforce(troops)

    def end(self) -> None:
        curr_snap = self._snapshot()
        rew = self._compute_reward(self.snapshot_prev, curr_snap)
        self.memory['state'].append(self.prev_state)
        self.memory['action'].append(self.prev_action)
        self.memory['logprob'].append(self.prev_logprob)
        self.memory['reward'].append(rew)
        self.memory['done'].append(True)
        self.returns[self.current].append(rew)
        if self.memory['state']:
            self._ppo_update()
            self._save()
        for strat in self.strategies.values(): strat.end()
        print("\n[Strategy Summary]")
        for n in self.names:
            used = self.usage[n]
            avg = float(np.mean(self.returns[n])) if self.returns[n] else 0.0
            bias = self.bias[n]
            print(f"{n:>10}: used={used:3d}  avg_rew={avg:+.2f}  bias={bias:.2f}")

    def _compute_reward(self, prev: Dict[str,float], curr: Dict[str,float]) -> float:
        r = (curr['terr'] - prev['terr']) * 0.5 + (curr['forces'] - prev['forces']) * 0.1
        r += (curr['areas'] - prev['areas']) * 2.0
        if not prev['alive'] and curr['alive']: r += 2.0
        if prev['alive'] and not curr['alive']: r -= 5.0
        return float(np.tanh(r / self.config.reward_scale))