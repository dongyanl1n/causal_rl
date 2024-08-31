import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Callable
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from multidoorkey_env import MultiDoorKeyEnv
import torch.nn as nn
import gymnasium as gym
import torch
import argparse
import imageio
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train PPO on MultiDoorKeyEnv")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for PPO")
    parser.add_argument("--total_timesteps", type=float, default=2e6, help="Total timesteps for training")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-8x8-v0", help="Environment name")
    parser.add_argument("--goal_pos", type=str, default=None, help="Goal position for the agent")
    parser.add_argument("--load_model_name", type=str, default="None", help="Model to load")
    return parser.parse_args()

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
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# @profile
def main():
    args = parse_arguments()
    print(args)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    config = {
        "lr": args.lr,
        "total_timesteps": int(args.total_timesteps),
        "env_name": args.env_name,
        "load_model_name": args.load_model_name,
        "goal_pos": args.goal_pos
    }

    run = wandb.init(
        project="fourrooms", 
        entity="dongyanl1n",
        name=f"sb3-{args.env_name}_lr{args.lr}{'_load_'+args.load_model_name if args.load_model_name != 'None' else ''}",
        config=config,
        sync_tensorboard=True,
        tags=['FixedGoal'],
        dir="/network/scratch/l/lindongy/grid_blickets"
        )

    if args.env_name.startswith("MultiDoorKeyEnv"):
        _, size, n_keys, _ = args.env_name.split('-')
        size = int(size.split('x')[0])
        n_keys = int(n_keys.split('keys')[0])
        env = MultiDoorKeyEnv(n_keys=n_keys, size=size, render_mode='rgb_array')
        env = ImgObsWrapper(env)
    elif args.env_name == "MiniGrid-FourRooms-v0" and args.goal_pos is not None:
        # translate goal_pos string, eg. "1,1" to (1, 1)
        goal_pos = tuple(map(int, args.goal_pos.split(",")))
        env = gym.make(args.env_name, render_mode="rgb_array", goal_pos=goal_pos)
        env = ImgObsWrapper(env)
    else:
        env = gym.make(args.env_name, render_mode="rgb_array")
        env = ImgObsWrapper(env)

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
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

    # if args.load_model_name != "None":
    #     model.set_parameters(f"/network/scratch/l/lindongy/grid_blickets/models/{args.load_model_name}", device=device)
    # model = PPO.load(f"/network/scratch/l/lindongy/grid_blickets/models/{args.load_model_name}", device=device)

    # Integrate wandb with stable-baselines3
    model.learn(total_timesteps=config["total_timesteps"], 
                callback=WandbCallback(verbose=0))

    # Finish the wandb run
    run.finish()

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"Mean Reward: {mean_reward}, Standard Deviation: {std_reward}")

    # Save the model
    # model.save(f"/network/scratch/l/lindongy/grid_blickets/models/sb3ppo_{args.env_name}_lr{args.lr}")

    def generate_gif(env, model, num_episodes: int, gif_filename: str):
        frames = []
        r_eps = []
        for episode in range(num_episodes):
            r_ep = 0
            obs, _ = env.reset()
            done = False
            while not done:
                frames.append(env.render())
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(action)
                r_ep += r
            frames.append(env.render())
            r_eps.append(r_ep)
        print(f"Mean reward: {sum(r_eps) / num_episodes}")

        imageio.mimsave(gif_filename, frames, duration=100)

    # generate_gif(env, model, num_episodes=10, gif_filename=f'sb3ppo_{args.env_name}_lr{args.lr}_pretrained.gif')

    env.close()

    # move the log file to the correct directory
    import os
    os.system(f"mv runs/{run.id} /network/scratch/l/lindongy/grid_blickets/runs/{run.id}")

if __name__ == "__main__":
    main()
