# ai/random_ai.py
"""
RandomAI‑v2 – retains unpredictability but avoids reckless plays.

Keeps the “coin‑flip” flavour yet biases toward:
▪ safer initial claims,
▪ reinforcing borders more often,
▪ only attacking when odds ≥ 50 %,
▪ up to 4 conquests per turn,
▪ smarter freemove toward weakest border.
"""

import random
from typing import List, Tuple

class RandomAI:
    def __init__(self, player, game, world, **kwargs):
        self.player = player
        self.game   = game
        self.world  = world
        self.rng    = random.Random()

    # ------------- lifecycle stubs ------------------------------------ #
    def start(self):  pass
    def end(self):    pass
    def event(self, msg): pass

    # ------------- initial placement ---------------------------------- #
    def initial_placement(self, empty_list, remaining):
        if empty_list:
            if self.rng.random() < 0.7:
                # 70 % choose lowest‑degree unclaimed territory
                pick = min(empty_list, key=lambda t: len(t.connect))
            else:
                pick = self.rng.choice(empty_list)
        else:
            terrs = list(self.player.territories)
            pick  = self.rng.choice(terrs)
        return pick.name

    # ------------- reinforce ------------------------------------------ #
    def reinforce(self, troops: int):
        terrs   = list(self.player.territories)
        borders = [t for t in terrs
                   if any(n.owner and n.owner != self.player for n in t.connect)]
        if borders and self.rng.random() < 0.8:      # 80 % bias
            target = self.rng.choice(borders)
        else:
            target = self.rng.choice(terrs)
        return {target.name: troops}

    # ------------- attack --------------------------------------------- #
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        orders    = []
        conquest  = 0
        max_cons  = 4

        # helper: all “reasonable” attacks
        def candidate_attacks():
            for src in self.player.territories:
                if src.forces < 3:              # need 2 dice + 1 stay
                    continue
                for tgt in src.adjacent(friendly=False):
                    if (src.forces - 1) >= tgt.forces + 1:  # odds ≥ ~50 %
                        yield src, tgt

        while conquest < max_cons:
            cand = [p for p in candidate_attacks()]
            if not cand or self.rng.random() < 0.5:
                break
            # random among top‑N (N= min(5, len(cand)))
            cand.sort(key=lambda st: (st[0].forces - 1) / st[1].forces,
                       reverse=True)
            topN = cand[:max(1, min(5, len(cand)))]
            src, tgt = self.rng.choice(topN)

            def continue_fn(n_atk, n_def):
                # 40 % chance to keep rolling if still have troops >1
                return self.rng.random() < 0.4 and n_atk > 1

            def move_fn(n_atk):
                max_move = min(3, n_atk - 1)
                # 60 % pick half, else min rule
                if self.rng.random() < 0.6 and n_atk > 6:
                    return max(n_atk // 2, 3)
                return self.rng.randint(1, max_move)

            orders.append((src.name, tgt.name, continue_fn, move_fn))
            conquest += 1

            # optimistic update so we don't re‑pick same tgt
            tgt.owner  = self.player
            moved      = move_fn(src.forces)
            tgt.forces = moved
            src.forces -= moved

        return orders

    # ------------- freemove ------------------------------------------- #
    def freemove(self):
        if self.rng.random() >= 0.5:     # 50 % chance to freemove
            return None

        terrs = list(self.player.territories)
        rears = [t for t in terrs
                 if t.forces > 1 and
                    all(n.owner == self.player for n in t.connect)]
        if not rears:
            return None
        src = self.rng.choice(rears)

        # choose destination
        borders = [t for t in terrs
                   if any(n.owner and n.owner != self.player for n in t.connect)]
        if borders and self.rng.random() < 0.7:
            tgt = self.rng.choice(borders)
        else:
            friends = [n for n in src.connect if n.owner == self.player]
            if not friends:
                return None
            tgt = self.rng.choice(friends)

        move = src.forces - 1
        return (src.name, tgt.name, move)
