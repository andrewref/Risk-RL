# rl/ppo.py
"""Custom PPO trainer for PyRisk that supports
✓ GPU acceleration (CUDA if available)
✓ Checkpoint resume / continual learning
✓ Single checkpoint file that is overwritten each save
✓ Compatible with RiskEnv observation/action spaces

Run:
    python -m rl.ppo --timesteps 1000000               # fresh train
    python -m rl.ppo --resume checkpoints/ppo_risk.pt  # continue training

After training the model is saved to checkpoints/ppo_risk.pt and can be used by
`agents/ppo.py` for the ncurses game.
"""
from __future__ import annotations

import argparse, os, time, torch, numpy as np
from gym import spaces
from rl.env import RiskEnv
from typing import Tuple

# ───────────────────────── Utility: Pick device ────────────────────── #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────── Rollout Buffer ─────────────────────────── #
class RolloutBuffer:
    def __init__(self, size: int, obs_shape: Tuple[int, ...]):
        self.size = size
        self.ptr  = 0
        self.full = False
        self.obs  = torch.zeros((size, *obs_shape), dtype=torch.float32, device=DEVICE)
        self.actions  = torch.zeros(size, dtype=torch.long,  device=DEVICE)
        self.rewards  = torch.zeros(size, dtype=torch.float32, device=DEVICE)
        self.dones    = torch.zeros(size, dtype=torch.float32, device=DEVICE)
        self.logprobs = torch.zeros(size, dtype=torch.float32, device=DEVICE)
        self.values   = torch.zeros(size, dtype=torch.float32, device=DEVICE)
        self.adv      = torch.zeros(size, dtype=torch.float32, device=DEVICE)
        self.returns  = torch.zeros(size, dtype=torch.float32, device=DEVICE)

    def add(self, obs, act, rew, done, logp, val):
        self.obs[self.ptr]      = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        self.actions[self.ptr]  = act
        self.rewards[self.ptr]  = rew
        self.dones[self.ptr]    = done
        self.logprobs[self.ptr] = logp
        self.values[self.ptr]   = val
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True
            self.ptr  = 0

    def compute_adv(self, last_val: float, gamma: float, lam: float):
        gae = 0.0
        for i in reversed(range(self.size)):
            mask  = 1.0 - self.dones[i]
            delta = self.rewards[i] + gamma * last_val * mask - self.values[i]
            gae   = delta + gamma * lam * mask * gae
            self.adv[i]    = gae
            self.returns[i]= gae + self.values[i]
            last_val = self.values[i]
        # normalise advantages
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std().clamp_min(1e-6))

    def batches(self, batch_size: int):
        idx = torch.randperm(self.size, device=DEVICE)
        for start in range(0, self.size, batch_size):
            sl = idx[start:start+batch_size]
            yield (self.obs[sl], self.actions[sl], self.logprobs[sl],
                   self.adv[sl], self.returns[sl])

# ───────────────────── Actor‑Critic Network ────────────────────────── #
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        h = 256
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh())
        self.pi  = nn.Linear(h, act_dim)
        self.v   = nn.Linear(h, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)

# ───────────────────────── PPO Trainer Class ───────────────────────── #
class PPOTrainer:
    def __init__(self,
                 obs_space: spaces.Box,
                 act_space: spaces.Discrete,
                 lr=2.5e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=4, batch=512, rollout=4096,
                 ent_coef=0.01, val_coef=0.5):
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = act_space.n
        self.gamma   = gamma
        self.lam     = lam
        self.clip_eps= clip_eps
        self.epochs  = epochs
        self.batch   = batch
        self.ent_c   = ent_coef
        self.val_c   = val_coef

        self.net = ActorCritic(self.obs_dim, self.act_dim).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buf = RolloutBuffer(rollout, obs_space.shape)

    def policy(self, obs_t: torch.Tensor):
        logits, val = self.net(obs_t)
        dist = Categorical(logits=logits)
        act  = dist.sample()
        return act, dist.log_prob(act), dist.entropy(), val

    def select(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        act, logp, _, val = self.policy(obs_t)
        return act.item(), logp.item(), val.item()

    # ── training loop ──────────────────────────────────────────────── #
    def train(self, env: RiskEnv, total_steps: int, save_path="checkpoints/ppo_risk.pt", save_every=100_000):
        obs, _     = env.reset()
        ep_rewards = 0.0
        start_time = time.time()
        for step in range(1, total_steps+1):
            act, logp, val = self.select(obs)
            next_obs, rew, done, _, info = env.step(act)
            self.buf.add(obs, act, rew, done, logp, val)
            obs = next_obs
            ep_rewards += rew

            if done:
                obs, _ = env.reset()
                ep_rewards = 0.0

            if self.buf.full:
                with torch.no_grad():
                    _, last_val = self.net(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                self.buf.compute_adv(last_val.item(), self.gamma, self.lam)
                self._update()

            if step % save_every == 0:
                torch.save(self.net.state_dict(), save_path)
                print(f"[INFO] Saved checkpoint at step {step}")

        torch.save(self.net.state_dict(), save_path)
        print(f"Training complete ({(time.time()-start_time)/60:.1f} min). Final model saved → {save_path}")

    # ── PPO gradient update ─────────────────────────────────────────── #
    def _update(self):
        for _ in range(self.epochs):
            for obs_b, act_b, logp_b, adv_b, ret_b in self.buf.batches(self.batch):
                logits, val_pred = self.net(obs_b)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_b)
                ratio = (new_logp - logp_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                val_loss = (ret_b - val_pred).pow(2).mean()
                entropy  = dist.entropy().mean()

                loss = policy_loss + self.val_c*val_loss - self.ent_c*entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

# ──────────────────────────── CLI entry ────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--resume", type=str, default=None, help="Path to existing .pt to resume from")
    parser.add_argument("--save-every", type=int, default=100_000)
    args = parser.parse_args()

    env = RiskEnv()
    trainer = PPOTrainer(env.observation_space, env.action_space)

    if args.resume and os.path.isfile(args.resume):
        trainer.net.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        print(f"[INFO] Resumed weights from {args.resume}")

    trainer.train(env, total_steps=args.timesteps, save_path="checkpoints/ppo_risk.pt", save_every=args.save_every)