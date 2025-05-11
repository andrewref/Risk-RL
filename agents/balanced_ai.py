# agents/balanced_ai.py
"""
BalancedAI 3.0 – even-keeled strategist
───────────────────────────────────────
• CLAIM            – prefers expandable spots, but not obsessively.
• EXTRA TROOPS     – drops on weakest border (fallback: random).
• REINFORCEMENT    – 50 % weakest border, 50 % strongest border (if any).
• ATTACK           – cautious-but-active: up to 2 conquests early, 3 later;
                     needs ~65 % win odds (ratio ≥ 1.25, or 1.1 for
                     continent finish / elimination).
• FREEMOVE         – shifts surplus from safest interior to weakest border
                     or spearhead.
"""

import random
from collections import defaultdict
from typing import List, Tuple, Dict

class BalancedAI:
    # ───────────────── constructor ─────────────────
    def __init__(self, player, game, world, **kwargs):
        self.player, self.game, self.world = player, game, world
        self.rng = random.Random()

    # ───────────────── lifecycle (unused) ──────────
    def start(self):  pass
    def end(self):    pass
    def event(self, msg): pass

    # ───────────── initial placement ───────────────
    def initial_placement(self, empty_list, remaining):
        if empty_list:                              # claim
            # Choose territory with many unclaimed neighbours (good growth)
            return max(empty_list,
                       key=lambda t: sum(n.owner is None for n in t.connect)
                      ).name
        # extra troops  – place on weakest border (fallback random)
        owned = list(self.player.territories)
        pick  = self._weakest_border(owned) or self.rng.choice(owned)
        return pick.name

    # ───────────── reinforcement (50 / 50) ─────────
    def reinforce(self, troops: int) -> Dict[str, int]:
        owned = list(self.player.territories)
        wk = self._weakest_border(owned)
        sp = self._strongest_border(owned)
        if wk is None:
            return {self.rng.choice(owned).name: troops}
        if sp is None or sp == wk:
            return {wk.name: troops}
        half = troops // 2
        return {wk.name: half, sp.name: troops - half}

    # ───────────── attack (moderate) ───────────────
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        orders, conquests = [], 0
        owned_territories = list(self.player.territories)

        # Dynamic cap: 2 conquests early, 3 when we’re larger
        def max_cons():
            owned = len(owned_territories)
            return 3 if owned >= 20 else 2

        terr_cnt = defaultdict(int)
        for t in self.world.territories.values():
            if t.owner:
                terr_cnt[t.owner] += 1

        while conquests < max_cons():
            best_pair, best_score = None, -1.0
            for src in owned_territories:
                if src.forces < 4:
                    continue
                for tgt in src.adjacent(friendly=False):
                    ratio = (src.forces - 1) / tgt.forces
                    need  = 1.25                      # ≈65 % odds baseline

                    finishes = all(tt.owner == self.player or tt == tgt
                                   for tt in tgt.area.territories)
                    eliminates = terr_cnt[tgt.owner] <= 2
                    if finishes or eliminates:
                        need = 1.10                  # more willing

                    if ratio < need:
                        continue

                    score = ratio
                    if finishes:   score += 1.5
                    if eliminates: score += 2.0
                    if score > best_score:
                        best_pair, best_score = (src, tgt), score

            if not best_pair:
                break

            src, tgt = best_pair
            def cont(n_atk, n_def):
                return n_atk > n_def + 1 and len(orders) < max_cons()

            def move(n_atk):
                return max(2, min(3, n_atk // 2))

            orders.append((src.name, tgt.name, cont, move))
            conquests += 1

            # optimistic update
            tgt.owner, moved = self.player, move(src.forces)
            tgt.forces, src.forces = moved, src.forces - moved

        return orders

    # ───────────── freemove (support defence) ──────
    def freemove(self):
        owned = list(self.player.territories)
        rear  = [t for t in owned
                 if t.forces > 1 and
                    all(n.owner == self.player for n in t.connect)]
        if not rear:
            return None

        src  = max(rear, key=lambda t: t.forces)
        dest = self._weakest_border(owned) or self._strongest_border(owned)
        if not dest or dest == src:
            return None
        return (src.name, dest.name, src.forces - 1)

    # ───────────── helpers ─────────────────────────
    def _weakest_border(self, terrs):
        weakest, worst = None, float("inf")
        for t in terrs:
            enemies = [n for n in t.connect if n.owner and n.owner != self.player]
            if not enemies:
                continue
            strongest = max(enemies, key=lambda n: n.forces)
            ratio = t.forces / strongest.forces
            if ratio < worst:
                weakest, worst = t, ratio
        return weakest

    def _strongest_border(self, terrs):
        strongest, best = None, -1.0
        for t in terrs:
            enemies = [n for n in t.connect if n.owner and n.owner != self.player]
            if not enemies:
                continue
            weakest_enemy = min(enemies, key=lambda n: n.forces)
            ratio = t.forces / weakest_enemy.forces
            if ratio > best:
                strongest, best = t, ratio
        return strongest
