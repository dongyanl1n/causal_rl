import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import miniworld # Import the MiniWorld environment
print(f"Imported miniworld from {miniworld.__file__}")
from miniworld.wrappers import PyTorchObsWrapper
import torch
import torch.nn as nn
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Callable
import numpy as np

class Flatten(nn.Module):
    """
    Flatten layer, to flatten convolutional layer output
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


def log_episode_video(episode_obs, episode_number, caption=None):
    """
    Logs a video of the episode's observations to wandb.
    
    Args:
    episode_obs (torch.Tensor): Tensor of shape (T, 3, 60, 80) containing the episode's observations.
    episode_number (int): The number or identifier of the episode.
    """
    # obs is numpy array of shape (T, 3, 60, 80), dtype=np.uint8
    video = wandb.Video(episode_obs,
                        caption=caption)
    wandb.log({f"episode_{episode_number}_video": video})
    print(f"Video for episode {episode_number} has been logged to wandb.")

    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train PPO on MultiDoorKeyEnv")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for PPO")
    parser.add_argument("--total_timesteps", type=float, default=2e6, help="Total timesteps for training")
    parser.add_argument("--env_name", type=str, default="MiniWorld-PutNext-v0", help="Environment name")
    parser.add_argument("--load_model_name", type=str, default="None", help="Model to load")
    return parser.parse_args()


class MiniWorldFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            Flatten(),
            # Print(),
            nn.Linear(1120, 128),
            nn.LeakyReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main():
    args = parse_arguments()
    print(args)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # initialize wandb
    print("Initializing wandb...")
    config = {
        "lr": args.lr,
        "total_timesteps": int(args.total_timesteps),
        "env_name": args.env_name,
        "load_model_name": args.load_model_name,
    }
    wandb.init(
        project=f"{args.env_name}_sb3", 
        entity="dongyanl1n",
        name=f"lr{args.lr}{'_load_'+args.load_model_name if args.load_model_name != 'None' else ''}",
        config=config,
        sync_tensorboard=True,
        dir="/network/scratch/l/lindongy/blicket_objects_env/policy"
        )

    # Create the environment
    print("Creating environment...")
    # env = gym.make(args.env_name, render_mode='rgb_array') 
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=False, norm_reward=False)
    # env = VecTransposeImage(env)
    env = gym.make(args.env_name)
    env = PyTorchObsWrapper(env)
    # Check the observation space
    print(f"Observation space: {env.observation_space}")
    
    # Initialize the PPO agent with GPU support
    print("Initializing PPO agent...")
    policy_kwargs = dict(
        features_extractor_class=MiniWorldFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    model = PPO("CnnPolicy", 
                env, 
                # policy_kwargs=policy_kwargs,
                verbose=1, 
                learning_rate=linear_schedule(args.lr),
                device=device, 
                tensorboard_log="/network/scratch/l/lindongy/blicket_objects_env/policy/sb3")

    # Train the agent
    total_timesteps = int(args.total_timesteps)
    model.learn(total_timesteps=total_timesteps,
                callback=WandbCallback(verbose=0))
    
    
    # Save the trained model
    # model.save(f"ppo_{args.env_name}")

    # Test the trained agent, logging a video of the episode
    obs, _ = env.reset()  # Unpack the tuple returned by reset
    test_episodes = 5
    for i in range(test_episodes):
        episode_obs = []
        episode_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_obs.append(np.transpose(obs, (0, 2, 1)))
            episode_reward += reward
        episode_obs = np.stack(episode_obs)
        log_episode_video(episode_obs, i, caption=f"Testing Episode {i}, reward: {episode_reward}")
        print(f"Testing Episode {i} reward: {episode_reward}")

    wandb.finish()

    env.close()

if __name__ == "__main__":
    main()