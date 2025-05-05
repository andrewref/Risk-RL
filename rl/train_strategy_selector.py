"""rl/train_strategy_selector.py

Train a small PPO policy whose *only* job is to pick the best opening
strategy (Aggressive, Balanced, Defensive, Random) given the initial
Risk board.  The policy sees the raw RiskEnv observation at turn‑0 and
chooses an action ∈ {0..3}.  The `StrategySelectorEnv` then executes that
strategy for 7 turns and returns a reward (+1 gain / −1 no‑gain).

We also log every outcome to `StrategyTracker`, giving us a persistent
JSON file of per‑situation statistics that you can inspect or use later
for ε‑greedy heuristic play.

Usage (fresh training):
    python -m rl.train_strategy_selector --timesteps 50000

Resume training:
    python -m rl.train_strategy_selector --resume selector_ppo.zip --timesteps 20000
"""
from __future__ import annotations

import argparse, os, time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl.strategy_selector_env import StrategySelectorEnv
from rl.strategy_tracker      import StrategyTracker

# ------------------------------------------------------------------- #
#  Custom callback to log outcomes to StrategyTracker                 #
# ------------------------------------------------------------------- #
class TrackerCallback(BaseCallback):
    def __init__(self, tracker: StrategyTracker, verbose: int = 0):
        super().__init__(verbose)
        self.tracker = tracker

    def _on_step(self) -> bool:
        # Called after each rollout step; but we only care at episode end
        if self.locals.get("dones") is not None:
            for done, info, obs, action, reward in zip(self.locals["dones"],
                                                       self.locals["infos"],
                                                       self.locals["obs"],
                                                       self.locals["actions"],
                                                       self.locals["rewards"]):
                if done:
                    self.tracker.update(obs, int(action), float(reward))
        return True

# ------------------------------------------------------------------- #
def main(args):
    env      = StrategySelectorEnv()
    tracker  = StrategyTracker(epsilon=args.epsilon)

    if args.resume and os.path.isfile(args.resume):
        model = PPO.load(args.resume, env=env, device="auto")
        print(f"[INFO] Resumed model from {args.resume}")
    else:
        model = PPO("MlpPolicy", env,
                    n_steps=256,
                    batch_size=64,
                    gae_lambda=0.95,
                    gamma=0.99,
                    ent_coef=0.01,
                    learning_rate=3e-4,
                    clip_range=0.2,
                    verbose=1)

    callback = TrackerCallback(tracker)
    start    = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback)
    print(f"[INFO] Training finished in {(time.time()-start)/60:.1f} min")

    save_path = args.save or "selector_ppo.zip"
    model.save(save_path)
    print(f"[INFO] Model saved → {save_path}")

# ------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=50000,
                   help="Total training timesteps")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to existing PPO zip to resume")
    p.add_argument("--save",   type=str, default="selector_ppo.zip",
                   help="Where to save the trained model")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="ε for ε‑greedy sampling in StrategyTracker")
    main(p.parse_args())
