from stable_baselines3 import PPO
from risk_env_multi import FourPlayerRiskEnv

MODEL_PATH = "checkpoints_multi/ppo_multi_final_500000.zip"

env = FourPlayerRiskEnv()
model = PPO.load(MODEL_PATH, env=env, device="cuda", print_system_info=True)

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, _ = env.step(action)
    env.render()
    
    if done or trunc:
        print("✅ Game over!")
        print("🏆 Winner:", env.world.current_player)
        break
