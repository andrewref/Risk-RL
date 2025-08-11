from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class RewardWeights:
    """Weights for individual reward components."""

    win: float = 1.0
    lose: float = -1.0
    delta_territory: float = 0.05
    delta_continent: float = 0.2
    attack_efficiency: float = 0.02  # (kills - losses)
    elimination: float = 0.5
    survival: float = 0.01
    clip_min: float = -1.0
    clip_max: float = 1.0


def compute_reward(
    prev: Dict[str, Dict[int, int]],
    curr: Dict[str, Dict[int, int]],
    events: Dict[str, Any],
    player_id: int | str,
    weights: RewardWeights,
) -> Tuple[float, Dict[str, float]]:
    """Compute shaped reward for ``player_id``.

    Parameters
    ----------
    prev, curr:
        Dictionaries containing at least ``territories`` and ``continents``
        entries mapping player IDs to counts.
    events:
        Event dictionary with optional keys: ``won_game_for``, ``lost_players``,
        ``kills``, ``losses``, ``eliminated``, and ``is_terminal``.
    player_id:
        Identifier of the acting player.
    weights:
        :class:`RewardWeights` controlling the contribution of each term.

    Returns
    -------
    total, components:
        Clipped total reward and the individual component values.
    """

    comps: Dict[str, float] = {}

    if events.get("won_game_for") == player_id:
        comps["win_loss"] = weights.win
    elif player_id in events.get("lost_players", []):
        comps["win_loss"] = weights.lose
    else:
        comps["win_loss"] = 0.0

    prev_t = prev.get("territories", {}).get(player_id, 0)
    curr_t = curr.get("territories", {}).get(player_id, 0)
    comps["delta_territory"] = weights.delta_territory * (curr_t - prev_t)

    prev_c = prev.get("continents", {}).get(player_id, 0)
    curr_c = curr.get("continents", {}).get(player_id, 0)
    comps["delta_continent"] = weights.delta_continent * (curr_c - prev_c)

    kills = events.get("kills", {}).get(player_id, 0)
    losses = events.get("losses", {}).get(player_id, 0)
    comps["attack_efficiency"] = weights.attack_efficiency * (kills - losses)

    comps["elimination"] = (
        weights.elimination if player_id in events.get("eliminated", []) else 0.0
    )
    comps["survival"] = (
        weights.survival if not events.get("is_terminal", False) else 0.0
    )

    total = sum(comps.values())
    total = max(min(total, weights.clip_max), weights.clip_min)
    return total, comps
