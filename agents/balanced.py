# ai/balanced_ai_v2.py
"""
BalancedAI‑v2 – stronger but still “moderate” bot.

– Dynamic aggression: 2 conquests early‑game, 4 once we’re big.
– Lower threshold (1.15×) for continent‑finishing or eliminations.
– Continues dice rolls while advantage holds, up to turn‑cap.
– Reinforcements: 70 % weakest border, 30 % spear‑head.
"""

import random
from collections import defaultdict
from typing import List, Tuple

class BalancedAI:
    # ------------ constructor ----------------------------------------- #
    def __init__(self, player, game, world, **kwargs):
        self.player = player
        self.game   = game
        self.world  = world
        self.rng    = random.Random()

    # ------------ lifecycle hooks ------------------------------------- #
    def start(self):  pass
    def end(self):    pass
    def event(self, msg): pass

    # ------------ initial placement ----------------------------------- #
    def initial_placement(self, empty_list, remaining):
        if empty_list:   # claim phase – choose territory with many free neighbours
            pick = max(empty_list,
                       key=lambda t: sum(n.owner is None for n in t.connect))
        else:            # extra troops
            terrs = list(self.player.territories)
            pick  = self._weakest_border(terrs) or self.rng.choice(terrs)
        return pick.name

    # ------------ reinforcement --------------------------------------- #
    def reinforce(self, troops: int):
        terrs = list(self.player.territories)
        weakest   = self._weakest_border(terrs)
        spearhead = self._strongest_border(terrs)
        if weakest is None:
            # No borders – stack randomly
            return {self.rng.choice(terrs).name: troops}
        if spearhead is None or spearhead == weakest:   # ← add this line
            return {weakest.name: troops} 
        if spearhead is None:
            spearhead = weakest

        wk_share = int(troops * 0.7)
        sp_share = troops - wk_share
        return {weakest.name: wk_share, spearhead.name: sp_share}

    # ------------ attack ---------------------------------------------- #
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        orders = []
        conquests = 0
          # ----------- NEW: cache territories as a real list -------------
        terrs = list(self.player.territories)          # ← add this
        def owned_count():                             # ← and this helper
            return len(terrs)
        # ----------------------------------------------------------------

        def max_conquests():
            owned = owned_count()                      # use helper
            bonus = self.player.reinforcements - 3
            return 4 if owned >= 25 or bonus >= 5 else 2

        # fast owner→territory count
        terr_cnt = defaultdict(int)
        for t in self.world.territories.values():
            if t.owner:
                terr_cnt[t.owner] += 1

        while conquests < max_conquests():
            candidates = []

            for src in terrs:
                if src.forces < 3:
                    continue
                for tgt in src.adjacent(friendly=False):
                    base_ratio = (src.forces - 1) / tgt.forces
                    need_ratio = 1.3

                    finishes_area = all(
                        tt.owner == self.player or tt == tgt
                        for tt in tgt.area.territories
                    )
                    eliminates = terr_cnt[tgt.owner] <= 2

                    if finishes_area or eliminates:
                        need_ratio = 1.15   # more willing

                    if base_ratio < need_ratio:
                        continue

                    score = base_ratio
                    if finishes_area: score += 2.0
                    if eliminates:    score += 3.0
                    candidates.append((score, src, tgt))

            if not candidates:
                break

            _, src, tgt = max(candidates, key=lambda stt: stt[0])

            def continue_fn(n_atk, n_def, _mx=max_conquests()):
                # continue as long as advantage holds AND we haven't exceeded cap
                return n_atk > n_def + 1 and len(orders) < _mx

            def move_fn(n_atk):
                # move half (at least 2, at most 3)
                return max(2, min(3, n_atk // 2))

            orders.append((src.name, tgt.name, continue_fn, move_fn))
            conquests += 1

            # optimistic update so later logic knows we now own tgt
            tgt.owner  = self.player
            moved      = move_fn(src.forces)
            tgt.forces = moved
            src.forces -= moved

        return orders

    # ------------ freemove -------------------------------------------- #
    def freemove(self):
        terrs = list(self.player.territories)
        rear  = [t for t in terrs
                 if t.forces > 1 and
                    all(n.owner == self.player for n in t.connect)]
        if not rear:
            return None

        src = max(rear, key=lambda t: t.forces)
        dest = self._weakest_border(terrs) or self._strongest_border(terrs)
        if dest is None or dest == src:
            return None
        move = src.forces - 1
        return (src.name, dest.name, move)

    # ------------ helpers --------------------------------------------- #
    def _weakest_border(self, terrs):
        weakest, worst_ratio = None, float("inf")
        for t in terrs:
            enemies = [n for n in t.connect if n.owner and n.owner != self.player]
            if not enemies:
                continue
            strongest = max(enemies, key=lambda n: n.forces)
            ratio = t.forces / strongest.forces
            if ratio < worst_ratio:
                weakest, worst_ratio = t, ratio
        return weakest

    def _strongest_border(self, terrs):
        best, best_ratio = None, -1.0
        for t in terrs:
            enemies = [n for n in t.connect if n.owner and n.owner != self.player]
            if not enemies:
                continue
            weakest = min(enemies, key=lambda n: n.forces)
            ratio = t.forces / weakest.forces
            if ratio > best_ratio:
                best, best_ratio = t, ratio
        return best
