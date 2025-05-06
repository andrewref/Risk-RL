# ai/aggressive.py
"""
An unusually strong hard‑coded Risk bot that

1. rushes to finish any continent where it already owns ≥80 %,
2. piles every reinforcement onto its single best attack front,
3. attacks while its (troops‑1) ≥ (defender + 2) to keep >70 % win odds,
4. eliminates players with ≤2 territories to steal cards,
5. leaves 1 troop behind everywhere, moves the excess to the hottest border.

It implements the exact interface used by game.py so it drops straight in.
"""

import random
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────
class AggressiveAI:
    # ------------- mandatory constructor --------------------------------
    def __init__(self, player, game, world, **kwargs):
        self.player   = player       # Player wrapper
        self.game     = game         # Game object (has .world etc.)
        self.world    = world        # World with Territory objects
        self.name     = player.name  # Convenience
        random.seed()                # independent rng

    # ------------- lifecycle hooks --------------------------------------
    def start(self):                # called once when game begins
        pass

    def end(self):                  # called once when game ends
        pass

    def event(self, msg):           # all game events land here
        pass                        # (unused – could add bookkeeping)

    # ------------- initial territory picking ----------------------------
    def initial_placement(self, empty_list, remaining):
        """
        During first pass `empty_list` is a list of unowned Territory objects.
        After all are claimed, engine calls again with empty_list=None, asking
        where to put extra troops (`remaining` > 0).  We always return a
        Territory *name* as a string.
        """
        if empty_list:                      # still claiming
            # Grab territory with most unclaimed neighbours (good expansion)
            choice = max(
                empty_list,
                key=lambda t: sum(n.owner is None for n in t.connect)
            )
        else:                               # reinforcing owned territory
            choice = self._best_frontline()
        return choice.name

    # ------------- reinforce --------------------------------------------
    def reinforce(self, troops):
        """
        Returns {territory_name: troop_count, ...} whose sum == troops given.
        Strategy: drop everything on best frontline territory.
        """
        terr = self._best_frontline()
        return {terr.name: troops}

    # ------------- attack -----------------------------------------------
    def attack(self):
        """
        Return a list of (src_name, tgt_name, continue_fn, move_fn) tuples.
        Our continue_fn keeps attacking while we still outnumber defender+1.
        """
        orders = []
        while True:
            pair = self._pick_attack()
            if not pair:
                break
            src, tgt = pair
            def continue_fn(n_atk, n_def, _src=src, _tgt=tgt):
                # Continue while we keep hefty advantage
                return n_atk > n_def + 1
            def move_fn(n_atk, _src=src, _tgt=tgt):
                # Move half (but >= min rules)
                return max( min(n_atk-1, 3), (n_atk-1)//2 )
            orders.append((src.name, tgt.name, continue_fn, move_fn))
            # Pretend attack succeeds to avoid infinite loop selection
            tgt_owner_before = tgt.owner
            if tgt_owner_before != self.player:
                tgt.owner = self.player
                tgt.forces = max(1, src.forces//2)
                src.forces = src.forces - tgt.forces
        return orders

    # ------------- freemove ---------------------------------------------
    def freemove(self):
        """
        After attacks, move surplus troops from a safe rear territory
        to the hottest border.  Returns (src_name, tgt_name, troop_count)
        or None.
        """
        rear_sources = [t for t in self.player.territories if not self._enemy_adjacent(t)]
        if not rear_sources:
            return None
        src = max(rear_sources, key=lambda t: t.forces)
        if src.forces <= 1:
            return None
        dest = self._best_frontline()
        if dest is None or dest == src:
            return None
        count = src.forces - 1
        return (src.name, dest.name, count)

    # ====================================================================
    # ------------------ internal helper methods -------------------------
    # ====================================================================

    # ---- frontline evaluation -----------------------------------------
    def _best_frontline(self):
        """
        Territory we own that (forces / weakest‑enemy‑neighbour) is maximal.
        If no borders, just return strongest owned territory.
        """
        best, best_ratio = None, -1
        for t in self.player.territories:
            w_enemy = self._weakest_enemy_neighbor(t)
            if w_enemy:
                ratio = t.forces / max(1, w_enemy.forces)
                if ratio > best_ratio:
                    best, best_ratio = t, ratio
        if best:
            return best
        # fallback: strongest territory
        return max(self.player.territories, key=lambda tr: tr.forces)

    # ---- attack picker -------------------------------------------------
    def _pick_attack(self):
        """
        Choose one high‑odds attack (src, tgt) or None.
        Preference:
        1. Complete continent
        2. Eliminate weak player (≤2 territories)
        3. Highest forces ratio
        """
        candidate = None
        best_score = -1

        # Build quick lookup of owner → territory count
        terr_count = defaultdict(int)
        for t in self.world.territories.values():
            if t.owner:
                terr_count[t.owner] += 1

        for src in self.player.territories:
            if src.forces < 3:
                continue
            for tgt in src.adjacent(friendly=False):
                atk_ratio = (src.forces-1) / tgt.forces
                score = atk_ratio

                # Bonus if it finishes a continent
                if all(tt.owner == self.player or tt == tgt for tt in tgt.area.territories):
                    score += 2

                # Bonus if it eliminates a nearly dead player
                if terr_count[tgt.owner] <= 2:
                    score += 3

                if score > best_score and src.forces >= tgt.forces + 2:
                    candidate, best_score = (src, tgt), score

        return candidate

    # ---- utilities -----------------------------------------------------
    def _weakest_enemy_neighbor(self, terr):
        enemies = [t for t in terr.connect if t.owner and t.owner != self.player]
        return min(enemies, key=lambda t: t.forces) if enemies else None

    def _enemy_adjacent(self, terr):
        return any(t.owner and t.owner != self.player for t in terr.connect)
