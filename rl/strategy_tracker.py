"""rl/strategy_tracker.py

A tiny utility class that keeps per‑strategy statistics across episodes
so you can *see* how each hard‑coded opening is doing and optionally use
those stats to bias future sampling.

Functions:
    • StrategyTracker.update(state_hash, strategy_id, reward)
    • StrategyTracker.best(strategy_id) – returns best score so far
    • StrategyTracker.sample() – epsilon‑greedy sampler you can call from
      your training loop instead of using an RL model for quick tests.

The tracker persists to JSON so you can resume long‑running experiments.
"""
from __future__ import annotations

import json, os, math, random
from collections import defaultdict
from typing import Dict, List

STRAT_NAMES = ["Aggressive", "Balanced", "Defensive", "Random"]

class StrategyTracker:
    """Keeps win/loss counts and mean reward per strategy & situation."""

    def __init__(self, save_path: str = "strategy_stats.json", epsilon: float = 0.1):
        self.save_path = save_path
        self.epsilon   = epsilon  # for epsilon‑greedy sampling
        # nested dict:  situ_hash → {strategy_id → [count, sum_reward]}
        self.table: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        if os.path.isfile(save_path):
            try:
                self.table.update(json.load(open(save_path)))
            except Exception:
                print("[WARN] Could not load existing strategy stats – starting fresh")

    # ---------------------------------------------------------------- #
    def _hash_state(self, obs) -> str:
        """Hash a raw observation into a compact string key (coarse)."""
        # Example: just use owner distribution histogram – simple & stable.
        owners = obs[::2]  # every second entry is owner id
        hist   = [int((owners == i).sum()) for i in range(5)]  # 0‑4 owners
        return ",".join(map(str, hist))

    # ---------------------------------------------------------------- #
    def update(self, obs, strategy_id: int, reward: float):
        key = self._hash_state(obs)
        rec = self.table[key][strategy_id]
        rec[0] += 1
        rec[1] += reward
        self._save()

    # ---------------------------------------------------------------- #
    def best(self, obs) -> int:
        """Return the strategy id with the best average reward so far for this state."""
        key = self._hash_state(obs)
        if key not in self.table:
            return random.randrange(len(STRAT_NAMES))
        # compute mean reward
        best_id, best_score = 0, -math.inf
        for sid, (cnt, s) in self.table[key].items():
            mean = s / cnt if cnt else -math.inf
            if mean > best_score:
                best_id, best_score = sid, mean
        return best_id

    # ---------------------------------------------------------------- #
    def sample(self, obs) -> int:
        """ε‑greedy: with prob ε choose random, else choose best so far."""
        if random.random() < self.epsilon:
            return random.randrange(len(STRAT_NAMES))
        return self.best(obs)

    # ---------------------------------------------------------------- #
    def _save(self):
        try:
            json.dump(self.table, open(self.save_path, "w"))
        except Exception as e:
            print("[WARN] Could not save strategy stats:", e)
