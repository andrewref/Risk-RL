#!/usr/bin/env python3
"""
Plot strategy–probability traces produced by PPOAgent.

Usage
-----
python plot_strategy_probs.py [traces/strategy_probs.pkl]

The pickle file may hold either
• one episode   →  [(step, [p0, p1, p2, p3]), …]
• many episodes →  list dumped repeatedly with pickle.dump(..., 'ab')

This script keeps reading objects from the pickle until EOF, concatenates them
and draws a line-plot for the four strategies.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Helper to load *all* pickled objects in the file                           #
# --------------------------------------------------------------------------- #
def load_prob_file(p: Path):
    episodes = []
    with p.open("rb") as f:
        while True:
            try:
                episodes.append(pickle.load(f))   # read next object
            except EOFError:
                break
    return episodes


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    fname = Path(sys.argv[1] if len(sys.argv) > 1 else "traces/strategy_probs.pkl")
    if not fname.exists():
        sys.exit(f"[plot] File not found: {fname}")

    episodes = load_prob_file(fname)
    if not episodes:
        sys.exit(f"[plot] No data found in {fname}")

    # Flatten episodes → two lists: steps and prob-vectors
    steps, probs = zip(*[pt for ep in episodes for pt in ep])
    probs = np.asarray(probs)              # shape (N, 4)

    labels = ["Aggressive", "Defensive", "Balanced", "Random"]
    plt.figure(figsize=(10, 5))
    for i, lbl in enumerate(labels):
        plt.plot(steps, probs[:, i], label=lbl)

    plt.title("Strategy probabilities over time")
    plt.xlabel("Training step")
    plt.ylabel("π(strat)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
