import os
import logging
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

# ───────────────────────── networks ─────────────────────────
class ActorNet(nn.Module):
    def __init__(self, inp, n_act, hid=64, lr=1e-3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, n_act)
        )
        nn.init.uniform_(self.body[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.body[-1].bias, 0.0)
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return torch.softmax(self.body(x), dim=-1)

class CriticNet(nn.Module):
    def __init__(self, inp, hid=64, lr=1e-3):
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
    seg_len, gamma, eps_clip, k_epochs = 7, 0.99, 0.2, 4
    entropy_coef, eps_greedy = 0.01, 0.10

    def __init__(self, player, game, world, **_):
        self.player, self.game, self.world = player, game, world
        self.name = player.name

        # Define strategies
        self.strats = {
            'aggressive': AggressiveAI(player, game, world),
            'balanced': BalancedAI(player, game, world),
            'defensive': DefensiveAI(player, game, world),
            'random': RandomAI(player, game, world)
        }
        self.idx2str = list(self.strats.keys())
        self.cur_str = 'random'

        # Neural Networks
        self.actor = ActorNet(5, len(self.idx2str))
        self.critic = CriticNet(5)

        # Memory for PPO
        self.mem = {k: [] for k in ['s', 'a', 'lp', 'r', 'd']}
        self.step = 0
        self.prev_state = self.prev_action = self.prev_lp = None
        self.snap_prev = None

        # Per-game tallies
        self.count = {k: 0 for k in self.idx2str}
        self.rew = {k: [] for k in self.idx2str}
        self.punished = {k: 0 for k in self.idx2str}  # Track number of times strategy is punished

        self.model_path = 'ppo_model.pt'
        if os.path.exists(self.model_path):
            ck = torch.load(self.model_path, map_location='cpu')
            self.actor.load_state_dict(ck['actor'], strict=False)
            self.critic.load_state_dict(ck['critic'], strict=False)

    # ───────── helpers ─────────
    def _feat(self):
        p = self.game.players[self.name]
        t_all = sum(pl.territory_count for pl in self.game.players.values() if pl.alive)
        f_all = sum(pl.forces for pl in self.game.players.values() if pl.alive)
        t_ratio = p.territory_count / t_all if t_all else 0
        f_ratio = p.forces / f_all if f_all else 0
        control = sum(1 for _ in p.areas) / len(self.world.areas)

        enemy, own = [], []
        for terr in self.world.territories.values():
            if terr.owner == p:
                for nb in terr.connect:
                    if nb.owner and nb.owner != p:
                        enemy.append(nb.forces)
                        own.append(terr.forces)
        threat = float(np.mean(enemy)) if enemy else 0
        weak = float(np.mean(own) / np.mean(enemy)) if (own and enemy and np.mean(enemy) > 0) else 0
        return torch.tensor([t_ratio, f_ratio, control, threat, weak], dtype=torch.float32)

    def _snap(self):
        p = self.game.players[self.name]
        return dict(terr=p.territory_count,
                    forces=p.forces,
                    areas=sum(1 for _ in p.areas),
                    alive=int(p.alive))

    def _reward(self, prev_snap, cur_snap):
        """
        Calculate the reward between the previous and current snapshots.
        prev_snap: The previous state snapshot
        cur_snap: The current state snapshot
        """
        r = (cur_snap['terr'] - prev_snap['terr']) * 0.5
        r += (cur_snap['forces'] - prev_snap['forces']) * 0.1
        r += (cur_snap['areas'] - prev_snap['areas']) * 2.0
        if not prev_snap['alive'] and cur_snap['alive']: r += 2  # Rebirth reward
        if prev_snap['alive'] and not cur_snap['alive']: r -= 5  # Death penalty
        return r

    # Punish strategies that performed poorly
    def _punish(self, punished_strat):
        self.punished[punished_strat] += 1
        if self.punished[punished_strat] > 2:  # Punish a strategy if it fails multiple times
            self.eps_greedy = 0.2
            self.punished[punished_strat] = 0  # Reset after handling punishment

    # ε-greedy choose
    def _choose(self, state):
        if np.random.rand() < self.eps_greedy:
            a = np.random.randint(len(self.idx2str))
            lp = torch.log(torch.tensor(1 / len(self.idx2str)))
            return a, lp
        dist = torch.distributions.Categorical(self.actor(state))
        a = dist.sample()
        return a.item(), dist.log_prob(a)

    # PPO update
    def _ppo(self):
        S = torch.stack(self.mem['s'])
        A = torch.tensor(self.mem['a'])
        OL = torch.stack(self.mem['lp']).detach()
        R, D = self.mem['r'], self.mem['d']

        ret, G = [], 0.0
        for r, dn in zip(reversed(R), reversed(D)):
            G = r + self.gamma * G * (1 - dn)
            ret.insert(0, G)
        ret = torch.tensor(ret)
        adv = ret - self.critic(S).squeeze().detach()

        for _ in range(self.k_epochs):
            dist = torch.distributions.Categorical(self.actor(S))
            NL = dist.log_prob(A)
            ratio = torch.exp(NL - OL)
            surr = torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv)
            lossA = -surr.mean()
            lossC = F.mse_loss(self.critic(S).squeeze(), ret)
            loss = lossA + 0.5 * lossC - self.entropy_coef * dist.entropy().mean()
            self.actor.opt.zero_grad()
            self.critic.opt.zero_grad()
            loss.backward()
            self.actor.opt.step()
            self.critic.opt.step()

        for k in self.mem:
            self.mem[k].clear()

    # save
    def _save(self):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()}, self.model_path)

    # ───────── engine callbacks ─────────
    def event(self, msg):
        for s in self.strats.values():
            if hasattr(s, 'event'):
                s.event(msg)

    def start(self):
        # reset per-game tallies
        for k in self.idx2str:
            self.count[k] = 0
            self.rew[k] = []
        self.step = 0
        self.cur_str = 'random'
        self.count[self.cur_str] += 1
        self.prev_state = None
        self.snap_prev = self._snap()
        for s in self.strats.values():
            s.start()

    def initial_placement(self, e, r):
        return self.strats[self.cur_str].initial_placement(e, r)

    def reinforce(self, troops):
        if self.step % self.seg_len == 0 and self.step > 0:
            st = self._feat()
            snap = self._snap()
            # store reward for previous segment
            rew = self._reward(self.snap_prev, snap)
            self.mem['s'].append(self.prev_state)
            self.mem['a'].append(self.prev_action)
            self.mem['lp'].append(self.prev_lp)
            self.mem['r'].append(rew)
            self.mem['d'].append(0)
            self.rew[self.cur_str].append(rew)

            # Punish strategies that performed poorly
            self._punish(self.cur_str)

            # choose next strategy
            a, lp = self._choose(st)
            self.cur_str = self.idx2str[a]
            self.count[self.cur_str] += 1
            self.prev_state, self.prev_action, self.prev_lp = st, a, lp
            self.snap_prev = snap

        # initialize first segment state lazily
        if self.prev_state is None:
            self.prev_state = self._feat()
            self.prev_action, self.prev_lp = 0, torch.log(torch.tensor(0.25))

        self.step += 1
        return self.strats[self.cur_str].reinforce(troops)

    def attack(self):
        return self.strats[self.cur_str].attack()

    def freemove(self):
        return self.strats[self.cur_str].freemove()

    def end(self):
        snap = self._snap()
        rew = self._reward(self.snap_prev, snap)
        self.mem['s'].append(self.prev_state)
        self.mem['a'].append(self.prev_action)
        self.mem['lp'].append(self.prev_lp)
        self.mem['r'].append(rew)
        self.mem['d'].append(1)
        self.rew[self.cur_str].append(rew)

        self._ppo()
        self._save()
        for s in self.strats.values():
            s.end()

        print("\n[Strategy Summary]")
        for k in self.idx2str:
            avg = np.mean(self.rew[k]) if self.rew[k] else 0.
            print(f"{k:>10}: used={self.count[k]:3d}  avg_rew={avg:+.2f}")
