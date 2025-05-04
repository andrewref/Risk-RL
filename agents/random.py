import random

class RandomAI:
    def __init__(self, player, game, world, **kwargs):
        self.player = player
        self.game = game
        self.world = world

    def start(self):
        pass

    def end(self):
        pass

    def event(self, msg):
        pass

    def initial_placement(self, empty_list, remaining):
        if empty_list:
            return random.choice(empty_list).name
        else:
            owned = list(self.player.territories)
            return random.choice(owned).name if owned else None

    def reinforce(self, troops):
        owned = list(self.player.territories)
        if not owned:
            return {}
        target = random.choice(owned)
        return {target.name: troops}

    def attack(self):
        orders = []
        for src in self.player.territories:
            if src.forces < 2:
                continue
            enemies = [t for t in src.adjacent(friendly=False)]
            if not enemies:
                continue
            tgt = random.choice(enemies)
            def cont_fn(n_atk, n_def):
                return n_atk > 1 and random.random() < 0.5
            def move_fn(n_atk):
                return max(1, (n_atk - 1) // 2)
            orders.append((src.name, tgt.name, cont_fn, move_fn))
            break  # Only 1 attack per turn
        return orders

    def freemove(self):
        srcs = [t for t in self.player.territories if t.forces > 1]
        if not srcs:
            return None
        src = random.choice(srcs)
        dsts = [t for t in self.player.territories if t != src]
        if not dsts:
            return None
        dst = random.choice(dsts)
        return (src.name, dst.name, src.forces - 1)
