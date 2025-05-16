# agents/aggressive_ai.py
"""
AggressiveAI – expert attacker, mediocre everywhere else
─────────────────────────────────────────────────────────
• CLAIM           → balanced behavior
• EXTRA TROOPS    → balanced behavior
• REINFORCEMENT   → balanced behavior
• ATTACK          → multi-step, high-odds assault; very aggressive
• FREEMOVE        → balanced behavior
"""

import random
from collections import defaultdict
from typing import List, Tuple, Dict
from AI.balanced_ai import BalancedAI  # delegate non-attack phases

class AggressiveAI:
    def __init__(self, player, game, world, **kwargs):
        self.player, self.game, self.world = player, game, world
        self.rng = random.Random()
        # use a BalancedAI for everything except attack
        self.normal_ai = BalancedAI(player, game, world)

    # ───────── lifecycle (unused) ─────────
    def start(self):       return self.normal_ai.start()
    def end(self):         return self.normal_ai.end()
    def event(self, msg):  return self.normal_ai.event(msg)

    # ───────── initial placement ─────────
    def initial_placement(self, empty_list, remaining):
        # delegate to balanced behavior
        return self.normal_ai.initial_placement(empty_list, remaining)

    # ───────── reinforcement ─────────
    def reinforce(self, troops: int) -> Dict[str, int]:
        # delegate to balanced behavior
        return self.normal_ai.reinforce(troops)

    # ───────── attack (professional) ─────────
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        """
        • Picks the best high-odds attack (≥70% win chance) repeatedly
          until either ⟶ max 3 conquests ⟶ source troop <4 ⟶ no suitable target.
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
                    if odds < 1.3:  # <≈70% win → skip
                        continue
                    score = odds
                    if all(tt.owner == self.player or tt == tgt
                           for tt in tgt.area.territories):
                        score += 2  # continent bonus
                    if terr_cnt[tgt.owner] <= 2:
                        score += 3  # elimination bonus
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
        # delegate to balanced behavior
        return self.normal_ai.freemove()
