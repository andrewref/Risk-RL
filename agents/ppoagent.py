# agents/ppoagent.py  — PPO meta‑agent that learns to pick from four hard‑coded strategies
# ---------------------------------------------------------------------------
# This version adds a **bias table** that is directly reinforced:
#   • Bad reward  → strategy bias is decreased (punished) → lower chance next time
#   • Good reward → strategy bias is increased (rewarded) → higher chance next time
# The bias values influence sampling *only* (not training), so the PPO gradient
# sees the true actor probabilities and remains stable.
# ---------------------------------------------------------------------------

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.aggressive_ai import AggressiveAI
from agents.balanced_ai   import BalancedAI
from agents.defensive_ai  import DefensiveAI
from agents.random_ai     import RandomAI

LOG = logging.getLogger("pyrisk.player.PPOAgent")

# ───────────────────────── networks ─────────────────────────
class ActorNet(nn.Module):
    def __init__(self, inp: int, n_act: int, hid: int = 64, lr: float = 1e-3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, n_act)
        )
        nn.init.uniform_(self.body[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.body[-1].bias, 0.0)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):                     # → raw logits
        return self.body(x)

class CriticNet(nn.Module):
    def __init__(self, inp: int, hid: int = 64, lr: float = 1e-3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.body(x)

# ───────────────────────── PPOAgent ─────────────────────────
class PPOAgent:
    seg_len, gamma, eps_clip, k_epochs = 7, 0.99, 0.20, 4
    entropy_coef, eps_greedy_base = 0.01, 0.10
    eps_greedy_max = 0.40
    bias_step, bias_min, bias_max = 0.10, 0.20, 3.0

    def __init__(self, player, game, world, **_):
        self.player, self.game, self.world = player, game, world
        self.name = player.name

        self.strats = {
            'aggressive': AggressiveAI(player, game, world),
            'balanced'  : BalancedAI (player, game, world),
            'defensive' : DefensiveAI(player, game, world),
            'random'    : RandomAI   (player, game, world)
        }
        self.idx2str = list(self.strats.keys())

        self.actor  = ActorNet(5, len(self.idx2str))
        self.critic = CriticNet(5)

        self.mem = {k: [] for k in ['s', 'a', 'lp', 'r', 'd']}
        self.step = 0
        self.cur_str = 'random'
        self.prev_state = self.prev_lp = None
        self.prev_action = self.idx2str.index(self.cur_str)
        self.snap_prev  = None

        self.count   = {k: 0 for k in self.idx2str}
        self.rew     = {k: [] for k in self.idx2str}
        self.punish  = {k: 0 for k in self.idx2str}
        self.eps_greedy = self.eps_greedy_base
        self.bias = {k: 1.0 for k in self.idx2str}

        self.model_path = 'ppo_model2.pt'
        if os.path.exists(self.model_path):
            ck = torch.load(self.model_path, map_location='cpu')
            self.actor.load_state_dict(ck['actor'], strict=False)
            self.critic.load_state_dict(ck['critic'], strict=False)

    def _feat(self):
        p = self.game.players[self.name]
        t_all = sum(pl.territory_count for pl in self.game.players.values() if pl.alive)
        f_all = sum(pl.forces          for pl in self.game.players.values() if pl.alive)
        t_ratio = p.territory_count / t_all if t_all else 0.0
        f_ratio = p.forces           / f_all if f_all else 0.0
        control = sum(1 for _ in p.areas) / len(self.world.areas)
        enemy, own = [], []
        for terr in self.world.territories.values():
            if terr.owner == p:
                for nb in terr.connect:
                    if nb.owner and nb.owner != p:
                        enemy.append(nb.forces); own.append(terr.forces)
        threat = float(np.mean(enemy)) if enemy else 0.0
        weak   = float(np.mean(own)/np.mean(enemy)) if (own and enemy and np.mean(enemy)>0) else 0.0
        return torch.tensor([t_ratio, f_ratio, control, threat, weak], dtype=torch.float32)

    def _snap(self):
        p = self.game.players[self.name]
        return dict(terr=p.territory_count, forces=p.forces,
                    areas=sum(1 for _ in p.areas), alive=int(p.alive))

    @staticmethod
    def _reward(a: dict, b: dict):
        r  = (b['terr']  - a['terr']) * 0.5
        r += (b['forces']- a['forces']) * 0.1
        r += (b['areas'] - a['areas']) * 2.0
        if not a['alive'] and b['alive']: r += 2.0
        if a['alive'] and not b['alive']: r -= 5.0
        return r

    def _choose(self, state):
        # exploration
        if np.random.rand() < self.eps_greedy:
            a  = np.random.randint(len(self.idx2str))
            lp = torch.log(torch.tensor(1.0/len(self.idx2str)))
            return a, lp

        logits = self.actor(state)
        plain_probs = torch.softmax(logits, dim=-1)
        # apply bias only for sampling
        bias_tensor = torch.tensor([self.bias[k] for k in self.idx2str], dtype=state.dtype)
        biased_probs = plain_probs * bias_tensor
        biased_probs = biased_probs / biased_probs.sum()
        dist_bias = torch.distributions.Categorical(biased_probs)
        a = dist_bias.sample()
        # store actor-only log-prob for PPO training
        dist_actor = torch.distributions.Categorical(plain_probs)
        lp = dist_actor.log_prob(a)
        return a.item(), lp

    def _adjust_bias(self, sname: str, rew: float):
        rew = max(-3.0, min(3.0, rew))  # clip reward
        if rew < 0:
            self.bias[sname] = max(self.bias_min, self.bias[sname] - self.bias_step)
            self.punish[sname] += 1
            if self.punish[sname] >= 3:
                self.eps_greedy = min(self.eps_greedy + 0.05, self.eps_greedy_max)
                self.punish[sname] = 0
        else:
            self.bias[sname] = min(self.bias_max, self.bias[sname] + self.bias_step)
            self.punish[sname] = 0
            self.eps_greedy = max(self.eps_greedy_base, self.eps_greedy - 0.01)

    def _ppo(self):
        S = torch.stack(self.mem['s'])
        A = torch.tensor(self.mem['a'])
        oldlp = torch.stack(self.mem['lp']).detach()
        R, D = self.mem['r'], self.mem['d']
        ret, G = [], 0.0
        for r, dn in zip(reversed(R), reversed(D)):
            G = r + self.gamma * G * (1 - dn)
            ret.insert(0, G)
        ret = torch.tensor(ret)
        adv = ret - self.critic(S).squeeze().detach()
        for _ in range(self.k_epochs):
            dist = torch.distributions.Categorical(torch.softmax(self.actor(S), dim=-1))
            newlp= dist.log_prob(A)
            ratio= torch.exp(newlp - oldlp)
            surr= torch.min(ratio * adv,
                           torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip)*adv)
            loss = -surr.mean() + 0.5*F.mse_loss(self.critic(S).squeeze(), ret)
            loss -= self.entropy_coef * dist.entropy().mean()
            self.actor.opt.zero_grad(); self.critic.opt.zero_grad()
            loss.backward(); self.actor.opt.step(); self.critic.opt.step()
        for k in self.mem: self.mem[k].clear()

    def _save(self):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()}, self.model_path)

    def event(self, msg):
        for s in self.strats.values():
            if hasattr(s, 'event'): s.event(msg)

    def start(self):
        for k in self.idx2str:
            self.count[k]=0; self.rew[k]=[]; self.punish[k]=0; self.bias[k]=1.0
        self.eps_greedy = self.eps_greedy_base
        self.step=0; self.cur_str='random'; self.prev_state=None
        self.prev_action=self.idx2str.index(self.cur_str)
        self.snap_prev=self._snap(); self.count[self.cur_str]+=1
        for s in self.strats.values(): s.start()

    def initial_placement(self,e,r): return self.strats[self.cur_str].initial_placement(e,r)
    def attack(self):               return self.strats[self.cur_str].attack()
    def freemove(self):             return self.strats[self.cur_str].freemove()

    def reinforce(self, troops):
        if self.step % self.seg_len==0 and self.step>0:
            snap_now=self._snap(); rew=self._reward(self.snap_prev,snap_now)
            self.mem['s'].append(self.prev_state); self.mem['a'].append(self.prev_action)
            self.mem['lp'].append(self.prev_lp); self.mem['r'].append(rew)
            self.mem['d'].append(0); self.rew[self.cur_str].append(rew)
            self._adjust_bias(self.cur_str, rew)
            st=self._feat(); a,lp=self._choose(st)
            self.cur_str=self.idx2str[a]; self.prev_state, self.prev_action, self.prev_lp = st,a,lp
            self.snap_prev=snap_now; self.count[self.cur_str]+=1
        if self.prev_state is None:
            self.prev_state=self._feat(); self.prev_lp=torch.log(torch.tensor(1.0/len(self.idx2str)))
        self.step+=1
        return self.strats[self.cur_str].reinforce(troops)

    def end(self):
        snap=self._snap(); rew=self._reward(self.snap_prev,snap)
        self.mem['s'].append(self.prev_state); self.mem['a'].append(self.prev_action)
        self.mem['lp'].append(self.prev_lp); self.mem['r'].append(rew); self.mem['d'].append(1)
        self.rew[self.cur_str].append(rew)
        if self.mem['s']: self._ppo()
        self._save();
        for s in self.strats.values(): s.end()
        if self.snap_prev is None: self.snap_prev=self._snap()
        print("\n[Strategy Summary]")
        for k in self.idx2str:
            avg = np.mean(self.rew[k]) if self.rew[k] else 0.0
            print(f"{k:>10}: used={self.count[k]:3d}  avg_rew={avg:+.2f}  bias={self.bias[k]:.2f}")
