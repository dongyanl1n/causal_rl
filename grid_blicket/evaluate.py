import os
import torch
import gymnasium as gym
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.envs import make_minigrid_envs
# from multidoorkey_env import MultiDoorKeyEnv
# from minigrid.wrappers import ImgObsWrapper
from Conspec.ConSpec import ConSpec
from arguments import get_args
from tqdm import tqdm
import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt

def obs_to_rgb(obs):
    # Assuming obs is a PyTorch tensor of shape (C, H, W)
    obs = obs.cpu().numpy()
    obs = (obs * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
    return obs

def make_gif(frames, filename, duration=100):
    """
    Make a gif from a list of frames.
    
    :param frames: List of numpy arrays representing frames
    :param filename: Output filename for the gif
    :param duration: Duration of each frame in the gif, in milliseconds
    """
    images = []
    for frame in frames:
        fig, ax = plt.subplots()
        ax.imshow(frame)
        ax.axis('off')
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)

    imageio.mimsave(filename, images, duration=duration)
    print(f"Gif saved as {filename}")

def evaluate(actor_critic, conspec, env, device, num_episodes=100):
    actor_critic.eval()
    conspec.encoder.eval()
    conspec.prototypes.prototypes.eval()

    episode_rewards = []
    episode_lengths = []
    for _ in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        
        # Initialize recurrent hidden states
        recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
        masks = torch.ones(1, 1).to(device)
        
        while not done:
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = actor_critic.act(
                    torch.transpose(obs, 3, 1).to(device),
                    recurrent_hidden_states,
                    masks,
                    deterministic=True
                )
            
            obs, reward, done, infos = env.step(action)
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    breakpoint
                    episode_rewards.append(info['episode']['r'])
                    episode_lengths.append(info['episode']['l'])

            # Update masks
            masks = torch.tensor([[0.0] if done else [1.0]], dtype=torch.float32).to(device)
    return sum(episode_rewards) / len(episode_rewards), sum(episode_lengths) / len(episode_lengths)

def record(actor_critic, conspec, env, device, num_episodes=10):
    actor_critic.eval()
    conspec.encoder.eval()

    all_episode_frames = []

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        episode_frames = []
        
        # Initialize recurrent hidden states
        recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
        masks = torch.ones(1, 1).to(device)
        
        while not done:
            # Convert observation to RGB image
            frame = obs_to_rgb(obs[0])  # obs[0] because we have only one environment
            episode_frames.append(frame)
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = actor_critic.act(
                    torch.transpose(obs, 3, 1),
                    recurrent_hidden_states,
                    masks,
                    deterministic=True
                )
            
            obs, reward, done, infos = env.step(action)

            # Update masks
            masks = torch.tensor([[0.0] if done else [1.0]], dtype=torch.float32).to(device)

        all_episode_frames.append(episode_frames)

        # Create a gif for this episode
        make_gif(episode_frames, f"episode_{episode}.gif")

    # Create a gif with all episodes
    all_frames = [frame for episode in all_episode_frames for frame in episode]
    make_gif(all_frames, "all_episodes.gif")
    print(f"Recorded {num_episodes} episodes")


def main():
    args = get_args()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Set up environment
    env, _ = make_minigrid_envs(num_envs=1, env_name=args.env_name, seeds=[args.seed], device=device)

    # Create models
    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'observation_space': env.observation_space}
    ).to(device)
    
    conspec = ConSpec(args, env.observation_space, env.action_space, device)

    # Load checkpoint
    checkpoint_path = "/home/mila/l/lindongy/linclab_folder/linclab_users/conspec_ckpt/high_rit_high_ret/MultiDoorKeyEnv-8x8-2keys-v0-conspec-rec-PO-lr0.0005-seed42/model_checkpoint_epoch_9999.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dictionaries
    actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
    conspec.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    conspec.prototypes.prototypes.load_state_dict(checkpoint['prototypes'])
    print(f"Loaded model from {checkpoint_path}")

    # Evaluate
    # mean_reward, mean_length = evaluate(actor_critic, conspec, env, device, num_episodes=100)
    # print(f"Mean episode reward over 100 episodes: {mean_reward}")
    # print(f"Mean episode length over 100 episodes: {mean_length}")

    # Record gifs
    record(actor_critic, conspec, env, device, num_episodes=10)

if __name__ == "__main__":
    main()