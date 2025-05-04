# ai/random_ai.py
"""
A purely‑random (yet rules‑respecting) Risk bot.

Strategy (such as it is):
1.  **Initial claim / reinforcement** – pick territories at random.
2.  **Reinforce** – dump all troops on a random owned territory
    (50 % chance to choose a border, 50 % chance anywhere).
3.  **Attack** – keep picking random legal attacks while
    - attacker has ≥ 3 troops, and
    - a coin‑flip (50 %) says “keep going”.
    For each win we move the minimum (Risk rules) or a random legal count.
4.  **Freemove** – 30 % chance to move surplus troops from a random
    rear territory to a random friendly neighbour.

Because it’s unpredictable it can still surprise smarter agents and give
your PPO learner useful variety.
"""

import random
from typing import List, Tuple

class RandomAI:
    # ------------------------------------------------------------------ #
    #  Mandatory constructor                                             #
    # ------------------------------------------------------------------ #
    def __init__(self, player, game, world, **kwargs):
        self.player   = player
        self.game     = game
        self.world    = world
        self.rng      = random.Random()          # private RNG

    # ------------------------------------------------------------------ #
    #  Lifecycle hooks (optional bookkeeping – unused here)              #
    # ------------------------------------------------------------------ #
    def start(self):  pass
    def end(self):    pass
    def event(self, msg): pass

    # ------------------------------------------------------------------ #
# ai/random_ai.py  ── fix for initial_placement  ------------------------

    def initial_placement(self, empty_list, remaining):
        """
        During the claiming phase we pick from `empty_list`.
        Once every territory is owned (empty_list is None) we place extra
        troops on a random territory we already own.
        """
        if empty_list:                         # still claiming
            pick = self.rng.choice(empty_list)
        else:                                  # reinforcing
            terrs = list(self.player.territories)   # ← convert generator ➜ list
            pick  = self.rng.choice(terrs)
        return pick.name

   # ai/random_ai.py  ── patched reinforce()  -----------------------------

    def reinforce(self, troops: int):
        """
        Put all reinforcements on one random owned territory.
        50 % chance to prefer a border; otherwise anywhere.
        """
        terrs = list(self.player.territories)                 # ← NEW
        borders = [t for t in terrs                           # ← CHANGED
                   if any(n.owner and n.owner != self.player for n in t.connect)]

        if borders and self.rng.random() < 0.5:
            target = self.rng.choice(borders)
        else:
            target = self.rng.choice(terrs)
        return {target.name: troops}


    # ------------------------------------------------------------------ #
    #  Attack                                                            #
    # ------------------------------------------------------------------ #
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        """
        Returns a list of (src, tgt, continue_fn, move_fn) orders.
        We keep issuing random legal attacks until:
          • no more legal attacks, or
          • 50 % coin‑flip tells us to stop.
        """
        orders = []

        # helper to find all current legal attacks
        def legal_attacks():
            for src in self.player.territories:
                if src.forces < 3:            # Risk needs ≥3 to attack (keep 1)
                    continue
                for tgt in src.adjacent(friendly=False):
                    yield (src, tgt)

        while True:
            attacks = list(legal_attacks())
            if not attacks or self.rng.random() < 0.5:
                break

            src, tgt = self.rng.choice(attacks)

            def continue_fn(n_atk, n_def):
                # stop after this battle (single round) 50 % of the time
                return self.rng.random() < 0.5 and n_atk > 1

            def move_fn(n_atk):
                """
                Risk rules: must leave at least 1 behind in source and move
                1‑3 (or all‑1) troops into conquered territory.
                """
                max_move = min(3, n_atk - 1)
                return self.rng.randint(1, max_move)

            orders.append((src.name, tgt.name, continue_fn, move_fn))

            # Pretend the attack succeeded so we don’t endlessly pick same pair
            tgt.owner = self.player
            moved = move_fn(src.forces)
            tgt.forces = moved
            src.forces -= moved

        return orders

    # ------------------------------------------------------------------ #
    #  Free‑move stage                                                   #
    # ------------------------------------------------------------------ #
    def freemove(self):
        """
        30 % chance to move all but one troop from a random ‘rear’ territory
        (no adjacent enemies) to a random friendly neighbour.
        Return (src_name, tgt_name, troops) or None.
        """
        if self.rng.random() >= 0.3:
            return None

        rears = [t for t in self.player.territories
                 if t.forces > 1 and
                    all(n.owner == self.player for n in t.connect)]
        if not rears:
            return None

        src = self.rng.choice(rears)
        friends = [n for n in src.connect if n.owner == self.player]
        if not friends:
            return None
        tgt = self.rng.choice(friends)
        move = src.forces - 1
        return (src.name, tgt.name, move)
