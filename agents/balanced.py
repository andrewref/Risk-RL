# ai/balanced_ai.py
"""
BalancedAI – a middle‑of‑the‑road Risk bot.

▪ Reinforces its weakest border each turn.
▪ Attacks cautiously: requires ≥1.3× defender troops and stops after
  at most 2 successful conquests.
▪ Prefers attacks that complete a continent or finish off a player.
▪ Moves surplus from safe rears to bolster the weakest front.
"""

import random
from collections import defaultdict
from typing import List, Tuple

class BalancedAI:
    # ------------------------------------------------------------------ #
    #  Constructor                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, player, game, world, **kwargs):
        self.player = player
        self.game   = game
        self.world  = world
        self.rng    = random.Random()

    # ------------------------------------------------------------------ #
    #  Lifecycle hooks (unused)                                          #
    # ------------------------------------------------------------------ #
    def start(self):  pass
    def end(self):    pass
    def event(self, msg): pass

    # ------------------------------------------------------------------ #
    #  Initial placement                                                 #
    # ------------------------------------------------------------------ #
    def initial_placement(self, empty_list, remaining):
        if empty_list:                                            # claim
            # Prefer territories with many unclaimed neighbours
            pick = max(
                empty_list,
                key=lambda t: sum(n.owner is None for n in t.connect)
            )
        else:                                                     # reinforce
            terrs = list(self.player.territories)
            pick  = self._weakest_border(terrs) or self.rng.choice(terrs)
        return pick.name

    # ------------------------------------------------------------------ #
    #  Reinforcement                                                     #
    # ------------------------------------------------------------------ #
    def reinforce(self, troops: int):
        terrs = list(self.player.territories)
        target = self._weakest_border(terrs) or self.rng.choice(terrs)
        return {target.name: troops}

    # ------------------------------------------------------------------ #
    #  Attack                                                            #
    # ------------------------------------------------------------------ #
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        orders = []
        attacks_done = 0

        def legal_attacks():
            for src in self.player.territories:
                if src.forces < 3:
                    continue
                for tgt in src.adjacent(friendly=False):
                    if (src.forces - 1) >= 1.3 * tgt.forces:   # ≥1.3× rule
                        yield src, tgt

        # quick owner→territory‑count lookup
        terr_cnt = defaultdict(int)
        for t in self.world.territories.values():
            if t.owner:
                terr_cnt[t.owner] += 1

        while attacks_done < 2:          # maximum two conquests/turn
            choices = list(legal_attacks())
            if not choices:
                break

            # score each (src,tgt)
            def score(pair):
                src, tgt = pair
                s = (src.forces - 1) / tgt.forces     # base: force ratio
                # continent bonus
                if all(tt.owner == self.player or tt == tgt
                       for tt in tgt.area.territories):
                    s += 1.5
                # elimination bonus
                if terr_cnt[tgt.owner] <= 2:
                    s += 2.0
                return s

            src, tgt = max(choices, key=score)

            def continue_fn(n_atk, n_def):
                # one battle round only (stop after conquest)
                return False

            def move_fn(n_atk):
                # Move just enough to hold:  = min(max(2, n_atk//2), 3)
                return max(2, min(3, n_atk // 2))

            orders.append((src.name, tgt.name, continue_fn, move_fn))
            attacks_done += 1

            # Pretend conquest succeeded to keep internal logic simple
            tgt.owner  = self.player
            moved      = move_fn(src.forces)
            tgt.forces = moved
            src.forces -= moved

        return orders

    # ------------------------------------------------------------------ #
    #  Freemove                                                          #
    # ------------------------------------------------------------------ #
    def freemove(self):
        terrs = list(self.player.territories)
        rear = [t for t in terrs
                if t.forces > 1 and
                   all(n.owner == self.player for n in t.connect)]
        if not rear:
            return None

        src = max(rear, key=lambda t: t.forces)
        dest = self._weakest_border(terrs)
        if not dest or dest == src:
            return None
        move = src.forces - 1
        return (src.name, dest.name, move)

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _weakest_border(self, terrs):
        """
        Return owned territory on enemy border with lowest force‑ratio
        (our troops / strongest adjacent enemy).  If none, return None.
        """
        weakest, best_ratio = None, float("inf")
        for t in terrs:
            enemies = [n for n in t.connect if n.owner and n.owner != self.player]
            if not enemies:
                continue
            max_enemy = max(enemies, key=lambda n: n.forces)
            ratio = t.forces / max_enemy.forces
            if ratio < best_ratio:
                weakest, best_ratio = t, ratio
        return weakest
