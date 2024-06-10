import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Callable
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from env import MultiDoorKeyEnv
import torch.nn as nn
import gymnasium as gym
import torch
import argparse

parser = argparse.ArgumentParser(description="Train PPO on MultiDoorKeyEnv")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for PPO")
parser.add_argument("--total_timesteps", type=float, default=2e6, help="Total timesteps for training")
parser.add_argument("--n_keys", type=int, default=3, help="Number of keys in the environment")
parser.add_argument("--size", type=int, default=8, help="Size of the environment")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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
    "n_keys": args.n_keys
}

run = wandb.init(
    project="grid_blicket_env", 
    entity="dongyanl1n",
    name=f"lr{args.lr}_n_keys{args.n_keys}_size{args.size}",
    config=config,
    sync_tensorboard=True,
    dir="/network/scratch/l/lindongy/grid_blickets"
    )

# Create the environment
env = MultiDoorKeyEnv(n_keys=args.n_keys, size=args.size, render_mode='rgb_array')
env = ImgObsWrapper(env)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),  # Increase the feature dimension size
    net_arch=dict(pi=[256, 256], vf=[256, 256])  # Increase the size of the PPO model
)

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=linear_schedule(args.lr), 
    verbose=0,
    tensorboard_log=f"runs/{run.id}",
    device=device
)

# Integrate wandb with stable-baselines3
model.learn(total_timesteps=config["total_timesteps"], 
            callback=WandbCallback(verbose=0))

# Finish the wandb run
run.finish()

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean Reward: {mean_reward}, Standard Deviation: {std_reward}")

# Save the model
# model.save("ppo_gridblickets_minigrid")

# Close the environment
env.close()

# move the log file to the correct directory
import os
os.system(f"mv runs/{run.id} /network/scratch/l/lindongy/grid_blickets/runs/{run.id}")

