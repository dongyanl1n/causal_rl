import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import miniworld

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the environment
env = gym.make("MiniWorld-OneRoom-v0")
env = DummyVecEnv([lambda: env])

# Initialize the PPO agent with GPU support
model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, device=device)

# Train the agent
total_timesteps = 1000000
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save(f"ppo_MiniWorld-OneRoom-v0")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()