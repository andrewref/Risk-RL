# pyrisk/world.py
"""
PyRisk – self-contained functional API plus an OO convenience wrapper
suitable for Stable-Baselines3 self-play experiments.
"""

from __future__ import annotations
import random
import textwrap
from collections import defaultdict

# ── Board definition ──────────────────────────────────────────────────
CONNECT = textwrap.dedent("""\
Alaska--Northwest Territories--Alberta--Alaska
Alberta--Ontario--Greenland--Northwest Territories
Greenland--Quebec--Ontario--Eastern United States--Quebec
Alberta--Western United States--Ontario--Northwest Territories
Western United States--Eastern United States--Mexico--Western United States

Venezuala--Peru--Argentina--Brazil
Peru--Brazil--Venezuala

North Africa--Egypt--East Africa--North Africa
North Africa--Congo--East Africa--South Africa--Congo
East Africa--Madagascar--South Africa

Indonesia--Western Australia--Eastern Australia--New Guinea--Indonesia
Western Australia--New Guinea

Iceland--Great Britain--Western Europe--Southern Europe--Northern Europe--Western Europe
Northern Europe--Great Britain--Scandanavia--Northern Europe--Ukraine--Scandanavia--Iceland
Southern Europe--Ukraine

Middle East--India--South East Asia--China--Mongolia--Japan--Kamchatka--Yakutsk--Irkutsk--Kamchatka--Mongolia--Irkutsk
Yakutsk--Siberia--Irkutsk
China--Siberia--Mongolia
Siberia--Ural--China--Afghanistan--Ural
Middle East--Afghanistan--India--China

Mexico--Venezuala
Brazil--North Africa
Western Europe--North Africa--Southern Europe--Egypt--Middle East--East Africa
Southern Europe--Middle East--Ukraine--Afghanistan--Ural
Ukraine--Ural
Greenland--Iceland
Alaska--Kamchatka
South East Asia--Indonesia
""")

# ── Area bonuses (unused by our simple bots, but here for completeness) ──
AREAS = {
    "North America": (5, ["Alaska", "Northwest Territories", "Greenland",
                          "Alberta", "Ontario", "Quebec",
                          "Western United States", "Eastern United States",
                          "Mexico"]),
    "South America": (2, ["Venezuala", "Brazil", "Peru", "Argentina"]),
    "Africa":        (3, ["North Africa", "Egypt", "East Africa",
                          "Congo", "South Africa", "Madagascar"]),
    "Europe":        (5, ["Iceland", "Great Britain", "Scandanavia", "Ukraine",
                          "Northern Europe", "Western Europe", "Southern Europe"]),
    "Asia":          (7, ["Middle East", "Afghanistan", "India", "South East Asia",
                          "China", "Mongolia", "Japan", "Kamchatka", "Irkutsk",
                          "Yakutsk", "Siberia", "Ural"]),
    "Australia":     (2, ["Indonesia", "New Guinea",
                          "Eastern Australia", "Western Australia"]),
}

# ── build adjacency graph once ────────────────────────────────────────
_NEIGHBORS: dict[str, list[str]] = defaultdict(list)
_TERRITORIES: list[str] = []
def _build_graph() -> None:
    for line in CONNECT.strip().splitlines():
        vals = [t.strip() for t in line.split("--")]
        for a, b in zip(vals, vals[1:]):
            if b not in _NEIGHBORS[a]:
                _NEIGHBORS[a].append(b)
            if a not in _NEIGHBORS[b]:
                _NEIGHBORS[b].append(a)
    global _TERRITORIES
    _TERRITORIES = sorted(_NEIGHBORS.keys())

_build_graph()

# ── global game state ────────────────────────────────────────────────
_OWNER:    dict[str,int]   = {}
_TROOPS:   dict[str,int]   = {}
_PLAYERS:  list[str]       = []
_CUR:      int             = 0
_LAST:     str             = ""

# ── Functional API ──────────────────────────────────────────────────
def reset(n_players: int = 4) -> None:
    """Start a fresh n-player game with one troop on each territory."""
    global _OWNER, _TROOPS, _PLAYERS, _CUR, _LAST
    _PLAYERS = [f"AI_{i}" for i in range(n_players)]
    _CUR = 0
    _LAST = "Game reset"
    # deal round-robin
    deck = _TERRITORIES[:]
    random.shuffle(deck)
    _OWNER  = {t: idx % n_players for idx, t in enumerate(deck)}
    _TROOPS = {t: 1 for t in _TERRITORIES}

def my_territories() -> list[str]:
    return [t for t,p in _OWNER.items() if p == _CUR]

def get_neighbors(terr: str) -> list[str]:
    return _NEIGHBORS[terr]

def is_enemy(terr: str) -> bool:
    return _OWNER[terr] != _CUR

def troops(terr: str) -> int:
    return _TROOPS[terr]

def attack(att: str, tgt: str) -> bool:
    global _LAST
    if tgt not in _NEIGHBORS[att]:
        _LAST = "Invalid adjacency"
        return False
    if _OWNER[att] != _CUR or _OWNER[tgt] == _CUR:
        _LAST = "Invalid ownership"
        return False
    if _TROOPS[att] <= _TROOPS[tgt] + 1:
        _LAST = "Not enough troops"
        return False
    # success
    _OWNER[tgt] = _CUR
    move = max(1, _TROOPS[att] // 2)
    _TROOPS[att] -= move
    _TROOPS[tgt] = move
    prev = (_CUR - 1) % len(_PLAYERS)
    _LAST = f"{_PLAYERS[_CUR]} took {tgt} from {_PLAYERS[prev]}"
    return True

def owner(terr: str) -> int:
    return _OWNER[terr]

def game_over() -> bool:
    first = owner(_TERRITORIES[0])
    return all(owner(t) == first for t in _TERRITORIES)

def next_player() -> None:
    global _CUR
    _CUR = (_CUR + 1) % len(_PLAYERS)

def current_player() -> int:
    return _CUR

def get_map() -> str:
    lines = [f"{t:24s} troops={_TROOPS[t]:2d} owner=AI{_OWNER[t]}"
             for t in _TERRITORIES]
    return "\n".join(lines)

def last_event() -> str:
    return _LAST

# upper bound for action‐space size
n_actions: int = sum(len(v) for v in _NEIGHBORS.values())

# ── OO Wrapper ─────────────────────────────────────────────────────
__all__ = ["World"]

class World:
    """Tiny OO wrapper around the functional PyRisk API."""

    def __init__(self):
        reset()               # brand new 4-player game

    def reset(self):
        reset()

    def my_territories(self):      return my_territories()
    def get_neighbors(self, t):    return get_neighbors(t)
    def is_enemy(self, t):         return is_enemy(t)
    def troops(self, t):           return troops(t)
    def attack(self, a, b):        return attack(a, b)
    def owner(self, t):            return owner(t)
    def game_over(self):           return game_over()
    def next_player(self):         next_player()

    @property
    def current_player(self):      return current_player()
    @property
    def territories(self):         return _TERRITORIES
    @property
    def n_actions(self):           return n_actions

    def get_map(self):             return get_map()
    def last_event(self):          return last_event()
