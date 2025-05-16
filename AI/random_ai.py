# agents/random_ai.py
"""
RandomAI – pure coin-flip bot
────────────────────────────
• CLAIM / EXTRA-TROOPS :   pick any legal territory at random
• REINFORCE            :   dump all troops on a random owned territory
• ATTACK               :   choose a random legal (src , tgt) once, roll once
• FREEMOVE             :   50 % chance to move every spare troop to a random friend
"""

import random
from typing import List, Tuple, Callable

class RandomAI:
    def __init__(self, player, game, world, **_):
        self.player = player
        self.world  = world
        self.rng    = random.Random()

    # ───────── lifecycle stubs ─────────
    def start(self):  pass
    def end(self):    pass
    def event(self, msg):  pass

    # ───────── initial placement ─────────
    def initial_placement(self, empty_list, remaining):
        if empty_list:
            terr = self.rng.choice(empty_list)               # claim phase
        else:
            owned = list(self.player.territories)
            terr  = self.rng.choice(owned) if owned else None   # extra troops
        return terr.name if terr else None

    # ───────── reinforcement ─────────
    def reinforce(self, troops: int):
        owned = list(self.player.territories)
        terr  = self.rng.choice(owned)
        return {terr.name: troops}

    # ───────── attack (one random attack, one roll) ─────────
    def attack(self) -> List[Tuple[str, str, Callable, Callable]]:
        candid = []
        for src in self.player.territories:
            if src.forces < 3:
                continue
            for tgt in src.adjacent(friendly=False):
                candid.append( (src, tgt) )

        if not candid:
            return []

        src, tgt = self.rng.choice(candid)

        # stop after first dice roll
        def cont(*_): return False
        # minimum legal move (leave 1 behind)
        def move_fn(n_atk): return max(1, min(3, n_atk-1))

        return [(src.name, tgt.name, cont, move_fn)]

    # ───────── freemove (50 % chance) ─────────
    def freemove(self):
        if self.rng.random() < 0.5:
            return None

        owned = list(self.player.territories)
        rears = [t for t in owned
                 if t.forces > 1 and all(n.owner == self.player for n in t.connect)]
        if not rears:
            return None

        src  = self.rng.choice(rears)
        dsts = [n for n in src.connect if n.owner == self.player]
        if not dsts:
            return None

        dst  = self.rng.choice(dsts)
        return (src.name, dst.name, src.forces - 1)
