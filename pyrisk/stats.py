import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict

class StatsCollector:
    """Collect per-turn statistics and save plots for Risk games."""
    def __init__(self, game, run_id: str | None = None):
        self.game = game
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path("runs") / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.turn = 0
        self.data: Dict[str, Dict[str, list]] = {}
        self.attacks: Dict[str, Dict[str, int]] = {}
        self.cum_rewards: Dict[str, float] = {}
        self.has_reward = False

    # ------------------------------------------------------------------
    def record_event(self, msg):
        """Track combat outcomes from game events."""
        if not msg:
            return
        tag = msg[0]
        if tag == "conquer":
            atk = msg[1].name
            dfn = msg[2].name
            self.attacks.setdefault(atk, {"won": 0, "lost": 0})["won"] += 1
            self.attacks.setdefault(dfn, {"won": 0, "lost": 0})["lost"] += 1
        elif tag == "defeat":
            atk = msg[1].name
            dfn = msg[2].name
            self.attacks.setdefault(atk, {"won": 0, "lost": 0})["lost"] += 1
            self.attacks.setdefault(dfn, {"won": 0, "lost": 0})["won"] += 1

    # ------------------------------------------------------------------
    def record_reward(self, player_name: str, reward: float) -> None:
        """Optionally accumulate reward for specific players."""
        self.has_reward = True
        self.cum_rewards[player_name] = self.cum_rewards.get(player_name, 0.0) + float(reward)

    # ------------------------------------------------------------------
    def record_turn(self) -> None:
        """Capture stats for all players at the end of a turn."""
        for name, p in self.game.players.items():
            stats = self.data.setdefault(name, {
                "territories": [],
                "armies": [],
                "continents": [],
                "attacks_won": [],
                "attacks_lost": [],
                "reward": [],
            })
            stats["territories"].append(p.territory_count)
            stats["armies"].append(p.forces)
            stats["continents"].append(sum(1 for _ in p.areas))
            atk = self.attacks.get(name, {"won": 0, "lost": 0})
            stats["attacks_won"].append(atk["won"])
            stats["attacks_lost"].append(atk["lost"])
            stats["reward"].append(self.cum_rewards.get(name, 0.0))
        self.turn += 1

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Generate and save a plot of collected statistics."""
        if not self.data:
            return
        metrics = ["territories", "armies", "continents", "attacks_won", "attacks_lost"]
        if self.has_reward:
            metrics.append("reward")
        turns = range(self.turn)
        rows = len(metrics)
        fig, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows), sharex=True)
        if rows == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            for name, stats in self.data.items():
                ax.plot(turns, stats[metric], label=name)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.legend(loc="upper left", fontsize="small")
        axes[-1].set_xlabel("Turn")
        fig.tight_layout()
        outfile = self.run_dir / f"{self.game.game_id}.png"
        fig.savefig(outfile)
        plt.close(fig)
