from AI.major import AI
import random
import collections

class FirstAI(AI):
    """
    StupidAI: Plays randomly. Reinforces and attacks without strategy.
    Safe for training loops and avoids infinite attack issues.
    """

    def __init__(self, player, game, world):
        super().__init__(player, game, world)

    def start(self):
        pass

    def event(self, msg):
        pass

    def end(self):
        pass

    def initial_placement(self, empty, remaining):
        if empty:
            return random.choice(empty)
        else:
            return random.choice(list(self.player.territories))

    def reinforce(self, available):
        border = [t for t in self.player.territories if t.border]
        if not border:
            border = list(self.player.territories)
        result = collections.defaultdict(int)
        for _ in range(available):
            t = random.choice(border)
            result[t] += 1
        return result

    def attack(self):
        # Each territory attacks at most once to prevent endless attack loops
        attacked = set()
        for t in self.player.territories:
            if t in attacked:
                continue
            targets = [a for a in t.connect if a.owner != self.player and t.forces > a.forces]
            if targets:
                a = random.choice(targets)
                yield (t, a, None, None)
                attacked.add(t)

    def freemove(self):
        # Does not perform any troop movement
        return None
