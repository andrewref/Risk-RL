import random
from typing import List, Tuple

class RandomAI:
    def __init__(self, player, game, world, **kwargs):
        self.player = player
        self.game   = game
        self.world  = world
        self.rng    = random.Random()

    # ------------- lifecycle stubs ------------------------------------ #
    def start(self): pass
    def end(self): pass
    def event(self, msg): pass

    # ------------- initial placement ---------------------------------- #
    def initial_placement(self, empty_list, remaining):
        # Ensure randomness but bias toward safer initial claims
        if empty_list:
            if self.rng.random() < 0.7:
                # 70% chance to pick lowest degree territory for a safer claim
                pick = min(empty_list, key=lambda t: len(t.connect))
            else:
                # 30% random selection if no preference
                pick = self.rng.choice(empty_list)
        else:
            # If no empty territories, pick a random territory from owned ones
            terrs = list(self.player.territories)
            pick  = self.rng.choice(terrs)
        return pick.name

    # ------------- reinforce ------------------------------------------ #
    def reinforce(self, troops: int):
        terrs   = list(self.player.territories)
        borders = [t for t in terrs
                   if any(n.owner and n.owner != self.player for n in t.connect)]
        if borders and self.rng.random() < 0.8:  # 80% chance to reinforce borders
            target = self.rng.choice(borders)
        else:
            target = self.rng.choice(terrs)  # Reinforce a random territory if no borders
        return {target.name: troops}

    # ------------- attack --------------------------------------------- #
    def attack(self) -> List[Tuple[str, str, callable, callable]]:
        orders    = []
        conquest  = 0
        max_cons  = 4  # Allow up to 4 conquests per turn

        def candidate_attacks():
            for src in self.player.territories:
                if src.forces < 3:  # Need at least 3 troops to attack
                    continue
                for tgt in src.adjacent(friendly=False):
                    if (src.forces - 1) >= tgt.forces + 1:  # Odds ≥ ~50%
                        yield src, tgt

        while conquest < max_cons:
            cand = [p for p in candidate_attacks()]
            if not cand or self.rng.random() < 0.5:
                break  # Stop if no reasonable attack or random chance is low

            # Pick randomly from the top N (with the best odds)
            cand.sort(key=lambda st: (st[0].forces - 1) / st[1].forces, reverse=True)
            topN = cand[:max(1, min(5, len(cand)))]  # Select the best N candidates
            src, tgt = self.rng.choice(topN)

            def continue_fn(n_atk, n_def):
                # 40% chance to keep rolling if still have troops left
                return self.rng.random() < 0.4 and n_atk > 1

            def move_fn(n_atk):
                max_move = min(3, n_atk - 1)
                # 60% chance to take half, else random between 1 and max_move
                if self.rng.random() < 0.6 and n_atk > 6:
                    return max(n_atk // 2, 3)
                return self.rng.randint(1, max_move)

            orders.append((src.name, tgt.name, continue_fn, move_fn))
            conquest += 1

            # Update target and source after move
            tgt.owner = self.player
            moved = move_fn(src.forces)
            tgt.forces = moved
            src.forces -= moved

        return orders

    # ------------- freemove ------------------------------------------- #
    def freemove(self):
        # 50% chance to make a free move
        if self.rng.random() >= 0.5:
            return None
