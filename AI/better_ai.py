from AI.major import AI
import random
import collections

class BetterAI(AI):
    """
    BetterAI: Prioritizes a specific continent and reinforces it.
    Picks targets with higher confidence based on force advantages.
    """
    def __init__(self, player, game, world):
        super().__init__(player, game, world)

    def start(self):
        # Randomize continent (area) priority at the start of the game
        self.area_priority = list(self.world.areas)
        random.shuffle(self.area_priority)

    def event(self, msg):
        # Placeholder to avoid AttributeError during gameplay events
        pass

    def end(self):
        # Called at the end of the game – currently a no-op
        pass

    def priority(self):
        border_territories = [t for t in self.player.territories if t.border]
        if not border_territories:
            return list(self.player.territories)

        # Rank by area priority
        sorted_by_priority = sorted(border_territories, key=lambda x: self.area_priority.index(x.area.name))
        top_area = sorted_by_priority[0].area
        return [t for t in sorted_by_priority if t.area == top_area]

    def initial_placement(self, empty, available):
        if empty:
            # Choose based on area priority
            empty_sorted = sorted(empty, key=lambda x: self.area_priority.index(x.area.name))
            return empty_sorted[0]
        else:
            return random.choice(self.priority())

    def reinforce(self, available):
        targets = self.priority()
        result = collections.defaultdict(int)
        while available > 0:
            result[random.choice(targets)] += 1
            available -= 1
        return result

    def attack(self):
        for t in self.player.territories:
            if t.forces > 1:
                adj_enemies = [a for a in t.connect if a.owner != t.owner and t.forces >= a.forces + 3]
                if len(adj_enemies) == 1:
                    yield (t.name, adj_enemies[0].name, lambda a, d: a > d, None)
                elif len(adj_enemies) > 1:
                    total = sum(a.forces for a in adj_enemies)
                    for adj in adj_enemies:
                        yield (t, adj, lambda a, d: a > d + total - adj.forces + 3, lambda a: 1)

    def freemove(self):
        interior = sorted([t for t in self.player.territories if not t.border], key=lambda x: x.forces)
        if interior:
            src = interior[-1]
            troops = src.forces - 1
            return (src, self.priority()[0], troops)
        return None
