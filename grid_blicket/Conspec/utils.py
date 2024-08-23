import torch
import numpy as np
import wandb

def log_highest_cos_sim_obs(self, obs_batch, prototype_cos_scores, prototype_number):
    '''logs the observation with the highest cosine similarity score to wandb.
    obs_batch: the observation batch retrieved; T, N, 3, 60, 80, where N is the number of success rollouts
    prototype_cos_scores: the cosine similarity scores; T, N
    '''
    cos_max, indices = torch.max(prototype_cos_scores, dim=0)  # t that maximizes the cosine similarity  # N
    high_cos_sim_obs = [obs_batch[t, i, :, :, :] for i, t in enumerate(indices)]  # len=N
    
    for i, (obs, cos_score) in enumerate(zip(high_cos_sim_obs, cos_max)):
        # Create a caption with the cosine similarity score
        caption = f"Update {self.update_counter}, Trajectory {i}, Cosine Similarity: {cos_score:.4f}"
        
        # Log to wandb with the caption
        wandb.log({
            f"prototype_{prototype_number}_highest_cos_sim": wandb.Image(
                obs.permute(1, 2, 0).cpu().numpy(),
                caption=caption
            )
        })

def log_episode_video(episode_obs, episode_number, caption=None):
    """
    Logs a video of the episode's observations to wandb.
    
    Args:
    episode_obs (torch.Tensor): Tensor of shape (T, 3, 60, 80) containing the episode's observations.
    episode_number (int): The number or identifier of the episode.
    """
    episode_obs = episode_obs.cpu().numpy()
    episode_obs = episode_obs.astype(np.uint8)
    video = wandb.Video(episode_obs,
                        caption=caption)
    wandb.log({f"episode_{episode_number}_video": video})
    print(f"Video for episode {episode_number} has been logged to wandb.")