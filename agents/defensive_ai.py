# ai/defensive_ai.py
"""
DefensiveAI‑v2 – stronger, still defence‑oriented.

Improvements
============
1. **Smarter reinforcement** – distribute troops across the two weakest
   borders (60 % / 40 %) instead of a single pile.
2. **Dynamic caution** – in the early game it will not attack unless the
   attacker has ≥ 2× defender.  Mid–late game (≥ 25 owned territories OR
   area bonus ≥ 5) threshold drops to 1.5× **but only** if the attack
   (a) finishes a continent **or** (b) eliminates a player.
3. **Up to two conquests** per turn (was one) if both satisfy the high
  ‑certainty rule above.
4. **Freemove** – funnel surplus from safe interiors toward both weakest
   borders rather than just one.
"""

import random
from collections import defaultdict
from typing import List, Tuple

class DefensiveAI:
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
        if empty_list:  # claim – favour low‑degree nodes (easy to defend)
            pick = min(empty_list, key=lambda t: len(t.connect))
        else:           # extra troops – weakest border
            terrs = list(self.player.territories)
            pick  = self._weakest_borders(terrs, k=1)[0] if terrs else None
            if pick is None:
                pick = self.rng.choice(terrs)
        return pick.name

    # ------------------------------------------------------------------ #
    #  Reinforcement                                                     #
    # ------------------------------------------------------------------ #
    def reinforce(self, troops: int):
        terrs = list(self.player.territories)
        wk1, wk2 = self._weakest_borders(terrs, k=2)
        if wk1 is None:          # no borders
            return {self.rng.choice(terrs).name: troops}

        if wk2 is None or wk2 == wk1:                 # ← modified condition
            return {wk1.name: troops} 

        t1 = int(troops * 0.6)
        t2 = troops - t1
        return {wk1.name: t1, wk2.name: t2}

    # ------------------------------------------------------------------ #
    #  Attack                                                            #
    # ------------------------------------------------------------------ #
      
      
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        orders   = []
        conquests = 0
        max_cons  = 2

        # --------- NEW: cache territories once instead of generator ----
        terrs = list(self.player.territories)        # convert generator ➜ list
        owned = len(terrs)                           # use cached list length
        # ----------------------------------------------------------------

        # owner → territory count
        terr_cnt = defaultdict(int)
        for t in self.world.territories.values():
            if t.owner:
                terr_cnt[t.owner] += 1

        bonus    = self.player.reinforcements - 3    # +areas, +cards
        mid_game = owned >= 25 or bonus >= 5

        while conquests < max_cons:
            best = None
            best_score = -1

            for src in terrs:
                if src.forces < 4:
                    continue
                for tgt in src.adjacent(friendly=False):
                    ratio = (src.forces - 1) / tgt.forces
                    need  = 2.0                        # default ≥2×

                    finishes = all(tt.owner == self.player or tt == tgt
                                   for tt in tgt.area.territories)
                    eliminates = terr_cnt[tgt.owner] <= 2

                    if mid_game and (finishes or eliminates):
                        need = 1.5                    # relaxed threshold

                    if ratio < need:
                        continue

                    score = ratio
                    if finishes:   score += 2
                    if eliminates: score += 3
                    if score > best_score:
                        best_score = score
                        best       = (src, tgt)

            if best is None:
                break

            src, tgt = best

            def continue_fn(n_atk, n_def, cap=max_cons):
                # stop after conquest OR if cap reached
                return False

            def move_fn(n_atk):
                # leave 2 behind at least, move up to 3
                return max(2, min(3, n_atk // 2))

            orders.append((src.name, tgt.name, continue_fn, move_fn))
            conquests += 1

            # optimistic update
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
        rears = [t for t in terrs
                 if t.forces > 1 and
                    all(n.owner == self.player for n in t.connect)]

        if not rears:
            return None

        src = max(rears, key=lambda t: t.forces)
        wk1, wk2 = self._weakest_borders(terrs, k=2)
        dest = wk1 or wk2
        if dest is None or dest == src:
            return None

        move = src.forces - 1
        return (src.name, dest.name, move)

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _weakest_borders(self, terrs, k=2):
        """
        Return up to k border territories with the worst force ratio
        (our troops / strongest adjacent enemy).
        """
        borders = []
        for t in terrs:
            enemies = [n for n in t.connect if n.owner and n.owner != self.player]
            if not enemies:
                continue
            strongest = max(enemies, key=lambda n: n.forces)
            ratio = t.forces / strongest.forces
            borders.append((ratio, t))
        borders.sort(key=lambda rt: rt[0])   # worst first
        picks = [b[1] for b in borders[:k]]
        # pad with None if fewer than k
        while len(picks) < k:
            picks.append(None)
        return picks
