import os
from stable_baselines3 import PPO
from procgen import ProcgenEnv

def main():
    # Retrieve the environment and agent names from environment variables
    env_name = "coinrun"
    agent_name = "PPO"

    # Create the environment
    env = ProcgenEnv(num_envs=16, env_name=env_name)

    if agent_name == "PPO":
        model = PPO("CnnPolicy", env, verbose=1)
        model.learn(total_timesteps=1000000)
        model.save(f"{env_name}_ppo")

    # You can add more agents by extending the if-else structure here.

if __name__ == "__main__":
    main()
