# evaluate_multi.py
from stable_baselines3 import PPO
from risk_env_multi import FourPlayerRiskEnv

MODEL_PATH = "checkpoints_multi/ppo_multi_loop10_YYYYMMDD_HHMM.zip"  # change

env   = FourPlayerRiskEnv()
model = PPO.load(MODEL_PATH, env=env, device="cuda")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, _ = env.step(action)
    env.render()
    if done or trunc:
        break
