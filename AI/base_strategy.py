# Risk‑RL/agents/base_strategy.py
"""
Base class for all hard‑coded Risk strategies.

Every child class *must* implement:
    choose_action(self, game_state) -> action

Parameters
----------
game_state : Any
    Whatever representation your game engine returns for the
    current player’s view of the board (e.g., territories, troops,
    phase, round number).

Returns
-------
action : Any
    A fully‑formed action object that your core Risk engine
    (e.g. pyrisk.world) will accept in its step() call.
"""

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Common interface that all strategies must follow."""

    @abstractmethod
    def choose_action(self, game_state):
        """
        Decide on a single legal action given the current game state.

        Child classes override this method with their own logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has not overridden choose_action()"
        )

    # (Optional) shared helper utilities
    # -----------------------------------
    def _territory_with_max_troops(self, territories):
        """Return id of territory with most troops (simple helper)."""
        return max(territories, key=lambda tid: territories[tid].troops)

    def _weakest_neighbor(self, territory, world):
        """Return neighbor territory with fewest troops, or None if none."""
        neighbors = world.get_neighbors(territory)
        enemy_neighbors = [
            n for n in neighbors if world.owner(n) != world.owner(territory)
        ]
        if not enemy_neighbors:
            return None
        return min(enemy_neighbors, key=lambda tid: world.troops(tid))
