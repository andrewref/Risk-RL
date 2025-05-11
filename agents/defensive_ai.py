# agents/defensive_ai.py
"""
DefensiveAI – fortress-only specialist
──────────────────────────────────────
• CLAIM / EXTRA TROOPS  → random (no grand plan)
• REINFORCEMENT         → strong: 60 % / 40 % to two weakest borders
• ATTACK                → almost never (only ultra-safe eliminations / finishes)
• FREEMOVE              → funnels surplus from interior to weakest border
"""

import random
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

class DefensiveAI:
    # ───────────────── constructor ─────────────────
    def __init__(self, player, game, world, **kwargs):
        self.player, self.game, self.world = player, game, world
        self.rng = random.Random()

    # ───────────────── lifecycle (unused) ──────────
    def start(self):  pass
    def end(self):    pass
    def event(self, msg): pass

    # ───────────── initial placement (plain) ───────
    def initial_placement(self, empty_list, remaining):
        """
        * Claim-phase: pick a random unowned territory.
        * Extra-troop phase: drop on a random owned territory.
        """
        if empty_list:                              # claim
            return self.rng.choice(empty_list).name
        owned = list(self.player.territories)       # extra troops
        return self.rng.choice(owned).name if owned else None

    # ───────────── reinforcement (strong) ──────────
    def reinforce(self, troops: int) -> Dict[str, int]:
        """
        60 % to weakest border, 40 % to second-weakest.
        If we have only one border, dump everything there.
        """
        owned = list(self.player.territories)
        wk1, wk2 = self._weakest_borders(owned, k=2)

        if wk1 is None:                                    # no borders – random
            return {self.rng.choice(owned).name: troops}

        if wk2 is None or wk2 == wk1:
            return {wk1.name: troops}

        t1 = int(troops * 0.6)
        return {wk1.name: t1, wk2.name: troops - t1}

    # ───────────── attack (ultra-cautious) ─────────
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        """
        Attack only if:
        • defender has ≤2 troops
        • we have ≥5 troops
        • AND the conquest either finishes a continent OR eliminates a player
        Executes **one** such conquest at most.
        """
        orders: List[Tuple[str, str, callable, callable]] = []
        owned = list(self.player.territories)

        # map owner → territory count
        terr_cnt = defaultdict(int)
        for t in self.world.territories.values():
            if t.owner:
                terr_cnt[t.owner] += 1

        for src in owned:
            if src.forces < 5:
                continue
            for tgt in src.adjacent(friendly=False):
                if tgt.forces > 2:
                    continue

                completes = all(tt.owner == self.player or tt == tgt
                                for tt in tgt.area.territories)
                eliminates = terr_cnt[tgt.owner] <= 1

                if not (completes or eliminates):
                    continue

                def cont(n_atk, n_def): return False
                def move(n_atk):        return max(2, min(3, n_atk // 2))
                orders.append((src.name, tgt.name, cont, move))
                return orders           # one-and-done
        return orders                   # likely empty – purely defensive

    # ───────────── freemove (border funnel) ────────
    def freemove(self):
        """Move surplus from safest interior stack to weakest border."""
        owned = list(self.player.territories)
        rear  = [t for t in owned
                 if t.forces > 1 and all(n.owner == self.player for n in t.connect)]
        if not rear:
            return None

        src  = max(rear, key=lambda t: t.forces)
        dest = self._weakest_border(owned)
        if dest is None or dest == src:
            return None
        return (src.name, dest.name, src.forces - 1)

    # ───────────── helper utilities ────────────────
    def _weakest_border(self, terrs):
        picks = self._weakest_borders(terrs, k=1)
        return picks[0] if picks else None

    def _weakest_borders(self, terrs, k=2):
        """
        Returns up to k border territories with the worst force ratio
        (our troops / strongest adjacent enemy troops).
        """
        
        borders = []
        for t in terrs:
            enemies = [n for n in t.connect if n.owner and n.owner != self.player]
            if not enemies:
                continue
            strongest = max(enemies, key=lambda n: n.forces)
            ratio = t.forces / strongest.forces
            borders.append((ratio, t))
        borders.sort(key=lambda rt: rt[0])        # ascending ⇒ weakest first
        picks = [b[1] for b in borders[:k]]
        while len(picks) < k:
            picks.append(None)
        return tuple(picks)
