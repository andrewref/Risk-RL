# rl/ppo.py
"""Light‑weight, self‑contained Proximal Policy Optimisation (PPO) engine
for the Risk RL project.

Why roll our own instead of importing Stable‑Baselines3?
=======================================================
• Easier to tweak / inspect internals.
• Zero external dependency beyond PyTorch + Gym.
• Full control of update schedule, entropy bonus, clipping, etc.

This file implements:
    • RolloutBuffer      – stores transitions, computes GAE advantages.
    • ActorCritic        – shared torso with separate policy & value heads.
    • PPOAgent           – selects actions, collects rollouts, performs SGD
                            updates with clipping.

Usage
-----
>>> from rl.env import RiskEnv
>>> from rl.ppo import PPOAgent
>>> env   = RiskEnv(seed=42)
>>> agent = PPOAgent(env.observation_space, env.action_space)
>>> agent.train(env, total_timesteps=1_000_000)
>>> agent.save("models/ppo_risk.pt")

The saved `.pt` file can later be loaded via `PPOAgent.load(path)` and
used in `eval.py`.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gym import spaces
from collections import deque
from typing import Tuple, List

# ───────────────────────────────── Rollout Buffer ──────────────────── #
class RolloutBuffer:
    def __init__(self, size: int, obs_shape: Tuple[int, ...], device: str):
        self.size      = size
        self.device    = device
        self.ptr       = 0
        self.full      = False
        self.observes  = torch.zeros((size, *obs_shape), dtype=torch.float32, device=device)
        self.actions   = torch.zeros(size, dtype=torch.long, device=device)
        self.rewards   = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones     = torch.zeros(size, dtype=torch.float32, device=device)
        self.logprobs  = torch.zeros(size, dtype=torch.float32, device=device)
        self.values    = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages= torch.zeros(size, dtype=torch.float32, device=device)
        self.returns   = torch.zeros(size, dtype=torch.float32, device=device)

    def add(self, obs, action, reward, done, logprob, value):
        self.observes[self.ptr]  = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.logprobs[self.ptr]  = logprob
        self.values[self.ptr]    = value
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True
            self.ptr  = 0

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        adv = 0.0
        for step in reversed(range(self.size)):
            mask  = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * last_value * mask - self.values[step]
            adv   = delta + gamma * lam * mask * adv
            self.advantages[step] = adv
            self.returns[step]    = adv + self.values[step]
            last_value = self.values[step]
        # normalise advantages
        adv_mean = self.advantages.mean()
        adv_std  = self.advantages.std().clamp_min(1e-6)
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get(self, batch_size: int):
        idx = np.random.permutation(self.size)
        for start in range(0, self.size, batch_size):
            slice_idx = idx[start:start+batch_size]
            yield (self.observes[slice_idx], self.actions[slice_idx], self.logprobs[slice_idx],
                   self.advantages[slice_idx], self.returns[slice_idx])

# ───────────────────────────────── Actor‑Critic Net ────────────────── #
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 256
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)
        return logits, value

# ───────────────────────────────────── PPO  ────────────────────────── #
class PPOAgent:
    def __init__(self,
                 obs_space: spaces.Box,
                 act_space: spaces.Discrete,
                 lr: float = 2.5e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_eps: float = 0.2,
                 epochs: int = 4,
                 batch_size: int = 256,
                 rollout_size: int = 2048,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 device: str = "cpu"):
        self.obs_dim   = int(np.prod(obs_space.shape))
        self.act_dim   = act_space.n
        self.gamma     = gamma
        self.lam       = lam
        self.clip_eps  = clip_eps
        self.epochs    = epochs
        self.batch_size= batch_size
        self.entropy_c = entropy_coef
        self.value_c   = value_coef
        self.device    = device

        self.net = ActorCritic(self.obs_dim, self.act_dim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer(rollout_size, obs_space.shape, device)

    # ────────────────────────────────────────────── API ────────────── #
    def select_action(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp   = dist.log_prob(action)
        return action.item(), logp.item(), value.item()

    def train(self, env, total_timesteps: int):
        obs, _ = env.reset()
        step_in_rollout = 0

        for t in range(total_timesteps):
            action, logp, value = self.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            self.buffer.add(obs, action, reward, done, logp, value)
            obs = next_obs
            step_in_rollout += 1

            if done:
                obs, _ = env.reset()

            if step_in_rollout >= self.buffer.size:
                with torch.no_grad():
                    _, last_val = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                self.buffer.compute_advantages(last_val.item(), self.gamma, self.lam)
                self._update()
                step_in_rollout = 0

    # ──────────────────────────────────────────────────────────────── #
    def _update(self):
        for _ in range(self.epochs):
            for obs_b, act_b, logp_b, adv_b, ret_b in self.buffer.get(self.batch_size):
                logits, values = self.net(obs_b)
                dist     = Categorical(logits=logits)
                new_logp = dist.log_prob(act_b)
                entropy  = dist.entropy().mean()

                ratio = (new_logp - logp_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss  = (ret_b - values).pow(2).mean()
                loss = policy_loss + self.value_c * value_loss - self.entropy_c * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

    # ───────────────────────────────────── util ────────────────────── #
    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path: str, obs_space: spaces.Box, act_space: spaces.Discrete, **kwargs):
        agent = cls(obs_space, act_space, **kwargs)
        agent.net.load_state_dict(torch.load(path, map_location=agent.device))
        return agent