import wandb
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from env import GridBlicketEnv
import torch.nn as nn
import gym
import torch
import argparse

parser = argparse.ArgumentParser(description="Train PPO on GridBlicketEnv")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO")
args = parser.parse_args()

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

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256),  # Increase the feature dimension size
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Increase the size of the PPO model
)

# env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
env = GridBlicketEnv(render_mode='rgb_array')
env = ImgObsWrapper(env)

wandb.init(project="grid_blicket_env", entity="dongyanl1n",name=f"lr{args.lr}")

# Initialize wandb callback
class WandbCallback(BaseCallback):
    def __init__(self):
        super(WandbCallback, self).__init__()

    def _on_step(self) -> bool:
        wandb.log({"reward": self.locals["rewards"].item()})
        return True

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=args.lr,  # Adjust the learning rate
    n_steps=2048,  # Number of steps to run for each environment per update
    batch_size=64,  # Minibatch size
    n_epochs=10,  # Number of epochs when optimizing the surrogate loss
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    clip_range=0.2,  # Clipping parameter, it can be a function
    ent_coef=0.01,  # Entropy coefficient
    verbose=1
)

# Integrate wandb with stable-baselines3
model.learn(total_timesteps=int(2e7), callback=WandbCallback())

# Finish the wandb run
wandb.finish()

