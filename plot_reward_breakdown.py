import json
import matplotlib.pyplot as plt
from pathlib import Path

with open("traces/reward_trace.json") as f:
    trace = json.load(f)

steps = [r["step"] for r in trace]
components = ["territory", "forces", "areas", "enemy_kills", "border_bonus", "win", "death"]

plt.figure(figsize=(12, 6))
for comp in components:
    values = [r[comp] for r in trace]
    plt.plot(steps, values, label=comp)

plt.title("Reward Component Breakdown per Step")
plt.xlabel("Step")
plt.ylabel("Reward Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/reward_breakdown.png")
plt.show()
