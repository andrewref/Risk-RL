#!/usr/bin/env python3
import logging
import random
import importlib
import re
import collections
import curses
import sys
import os
from pyrisk.game import Game
from pyrisk.world import CONNECT, MAP, KEY, AREAS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

LOG = logging.getLogger("pyrisk")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--nocurses", dest="curses", action="store_false", default=True,
                    help="Disable the ncurses map display")
parser.add_argument("--nocolor", dest="color", action="store_false", default=True,
                    help="Display the map without colors")
parser.add_argument("-l", "--log", action="store_true", default=False,
                    help="Write game events to a logfile")
parser.add_argument("-d", "--delay", type=float, default=0.1,
                    help="Delay in seconds after each action is displayed")
parser.add_argument("-s", "--seed", type=int, default=None,
                    help="Random number generator seed")
parser.add_argument("-g", "--games", type=int, default=1,
                    help="Number of rounds to play")
parser.add_argument("-w", "--wait", action="store_true", default=False,
                    help="Pause and wait for a keypress after each action")
parser.add_argument("players", nargs="+",
                    help="Names of the AI classes to use. May use 'ExampleAI*3' syntax.")
parser.add_argument("--deal", action="store_true", default=False,
                    help="Deal territories rather than letting players choose")

args = parser.parse_args()

NAMES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]

LOG.setLevel(logging.DEBUG)
if args.log:
    logging.basicConfig(filename="pyrisk.log", filemode="w")
elif not args.curses:
    logging.basicConfig()

if args.seed is not None:
    random.seed(args.seed)

# --------------------------------------------------------------------- #
#  Load AI classes                                                      #
# --------------------------------------------------------------------- #
player_classes = []
name_to_class = {}  # Map player name (e.g., "ALPHA") to class name (e.g., "PPOAgent")
for p in args.players:
    match = re.match(r"(\w+)?(\*\d+)?", p)
    if not match:
        continue
    name = match.group(1)
    filename = name[:-2].lower() + "_ai" if name.endswith("AI") else name.lower()
    count = int(match.group(2)[1:]) if match.group(2) else 1
    try:
        klass = getattr(importlib.import_module("agents." + filename), name)
        player_classes.extend([klass] * count)
    except Exception:
        print(f"Unable to import AI {name} from agents/{filename}.py")
        raise

# --------------------------------------------------------------------- #
#  Game runner                                                          #
# --------------------------------------------------------------------- #
kwargs = dict(
    curses=args.curses,
    color=args.color,
    delay=args.delay,
    connect=CONNECT,
    cmap=MAP,
    ckey=KEY,
    areas=AREAS,
    wait=args.wait,
    deal=args.deal,
)

def wrapper(stdscr, **kwargs):
    g = Game(screen=stdscr, **kwargs)
    for i, klass in enumerate(player_classes):
        name = NAMES[i]
        g.add_player(name, klass)
        name_to_class[name] = klass.__name__
    return g.play()

# Single game or multiple rounds
if args.games == 1:
    if args.curses:
        winner = curses.wrapper(wrapper, **kwargs)
    else:
        winner = wrapper(None, **kwargs)
    winner_class = name_to_class.get(winner, winner)
    print(f"\n🏆 The winner is: {winner_class}")
else:
    wins = collections.defaultdict(int)
    for j in range(args.games):
        kwargs['round'] = (j + 1, args.games)
        kwargs['history'] = wins
        victor = curses.wrapper(wrapper, **kwargs) if args.curses else wrapper(None, **kwargs)
        wins[victor] += 1

    print(f"Outcome of {args.games} games")
    for k in sorted(wins, key=lambda x: wins[x]):
        agent_class = name_to_class.get(k, "?")
        print(f"{k} [{agent_class}]:\t{wins[k]}")