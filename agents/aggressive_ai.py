# agents/aggressive_ai.py
"""
AggressiveAI – expert attacker, mediocre everywhere else
─────────────────────────────────────────────────────────
• CLAIM           → random unowned territory
• EXTRA TROOPS    → random owned territory
• REINFORCEMENT   → scatter troops randomly
• ATTACK          → multi-step, high-odds assault; very aggressive
• FREEMOVE        → pours surplus from safe interiors to best attack front
"""

import random
from collections import defaultdict
from typing import List, Tuple, Dict

class AggressiveAI:
    # ───────── constructor ─────────
    def __init__(self, player, game, world, **kwargs):
        self.player, self.game, self.world = player, game, world
        self.rng = random.Random()

    # ───────── lifecycle (unused) ─────────
    def start(self):  pass
    def end(self):    pass
    def event(self, msg): pass

    # ───────── initial placement ─────────
    def initial_placement(self, empty_list, remaining):
        # Phase-1 claim  → totally random
        if empty_list:
            return self.rng.choice(empty_list).name
        # Phase-2 extra troops → totally random
        owned = list(self.player.territories)
        return self.rng.choice(owned).name if owned else None

    # ───────── reinforcement ─────────
    def reinforce(self, troops: int) -> Dict[str, int]:
        """Scatter reinforcements randomly (no planning)."""
        owned = list(self.player.territories)
        alloc = defaultdict(int)
        for _ in range(troops):
            alloc[self.rng.choice(owned).name] += 1
        return alloc

    # ───────── attack (professional) ─────────
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        """
        • Picks the best high-odds attack (≥70 % win chance) repeatedly
          until either   ⟶ max 3 conquests   ⟶ source troop <4
          ⟶ no suitable target.
        """
        orders, conquests, CAP = [], 0, 3
        terr_cnt = defaultdict(int)
        for t in self.world.territories.values():
            if t.owner:
                terr_cnt[t.owner] += 1

        while conquests < CAP:
            best_pair, best_score = None, -1.0
            for src in self.player.territories:
                if src.forces < 4:
                    continue
                for tgt in src.adjacent(friendly=False):
                    odds = (src.forces - 1) / tgt.forces
                    if odds < 1.3:                # <≈70 % win → skip
                        continue
                    score = odds
                    if all(tt.owner == self.player or tt == tgt
                           for tt in tgt.area.territories):
                        score += 2               # continent bonus
                    if terr_cnt[tgt.owner] <= 2:
                        score += 3               # elimination bonus
                    if score > best_score:
                        best_pair, best_score = (src, tgt), score
            if not best_pair:
                break

            src, tgt = best_pair
            def cont(n_atk, n_def): return n_atk > n_def + 1
            def move(n_atk):        return max(2, min(3, n_atk - 1))

            orders.append((src.name, tgt.name, cont, move))
            conquests += 1

            # optimistic update so we don’t re-select same pair
            tgt.owner, moved = self.player, move(src.forces)
            tgt.forces, src.forces = moved, src.forces - moved

        return orders

    # ───────── freemove (support attack) ─────────
    def freemove(self):
        """Shift everything from the safest big stack to best attack front."""
        owned = list(self.player.territories)
        rear  = [t for t in owned
                 if t.forces > 1 and
                    all(n.owner == self.player for n in t.connect)]
        if not rear:
            return None
        src  = max(rear, key=lambda t: t.forces)
        dest = self._best_frontline(owned)
        if not dest or dest == src:
            return None
        return (src.name, dest.name, src.forces - 1)

    # ───────── helpers ─────────
    def _best_frontline(self, terrs):
        best, ratio = None, -1
        for t in terrs:
            enemy = self._strongest_enemy_neighbor(t)
            if not enemy:
                continue
            r = t.forces / enemy.forces
            if r > ratio:
                best, ratio = t, r
        return best

    def _strongest_enemy_neighbor(self, terr):
        enemies = [n for n in terr.connect if n.owner and n.owner != self.player]
        return max(enemies, key=lambda e: e.forces) if enemies else None



    # very very aggreisve only attackss 
    # defensivee only defensncee 
    # balanced  is not smart enough 
    # random do random things 
