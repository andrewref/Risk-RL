import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("traces/strategy_switch_log.json") as f:
    switch_log = json.load(f)

steps, strategies = zip(*switch_log)
strategy_ids = {"aggressive": 0, "defensive": 1, "balanced": 2, "random": 3}
strategy_nums = [strategy_ids[s] for s in strategies]

plt.figure(figsize=(10, 3))
plt.scatter(steps, strategy_nums, c=strategy_nums, cmap="tab10")
plt.yticks(list(strategy_ids.values()), list(strategy_ids.keys()))
plt.xlabel("Step")
plt.ylabel("Strategy")
plt.title("Strategy Switching Timeline")
plt.tight_layout()

Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/strategy_switch_timeline.png")
plt.show()
