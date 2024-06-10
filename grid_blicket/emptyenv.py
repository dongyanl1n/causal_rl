import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Callable
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import torch.nn as nn
import gymnasium as gym
import torch
import argparse

parser = argparse.ArgumentParser(description="Train PPO on EmptyEnv")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO")
parser.add_argument("--total_timesteps", type=float, default=2e6, help="Total timesteps for training")
parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0", help="Environment to train on")
args = parser.parse_args()
print(args)
    
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Define a custom CNN feature extractor for the MiniGrid environment
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

config = {
    "lr": args.lr,
    "total_timesteps": int(args.total_timesteps),
    "env": args.env
}

run = wandb.init(
    project="emptyenv", 
    entity="dongyanl1n",
    name=f"{args.env}_lr{args.lr}",
    config=config,
    sync_tensorboard=True,
    dir="/network/scratch/l/lindongy/emptyenv/"
    )

# Create the environment
env = gym.make(config['env'], render_mode="rgb_array")
env = ImgObsWrapper(env)

# Create the PPO model
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[128, 128], vf=[128, 128])
)

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=linear_schedule(args.lr),  # Adjust the learning rate
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    device=device
)



# Integrate wandb with stable-baselines3
model.learn(total_timesteps=config["total_timesteps"], 
            callback=WandbCallback(verbose=1))

# Finish the wandb run
run.finish()

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean Reward: {mean_reward}, Standard Deviation: {std_reward}")

# Save the model
# model.save("ppo_emptyenv_minigrid")

# Close the environment
env.close()

# move the log file to the correct directory
import os
os.system(f"mv runs/{run.id} /network/scratch/l/lindongy/emptyenv/runs/{run.id}")
