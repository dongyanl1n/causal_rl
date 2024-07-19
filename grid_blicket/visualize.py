import torch
import os 
import numpy as np
import matplotlib.pyplot as plt
from analysis_util import render_obs, plot_and_save_obs_image, save_episode_as_gif
    
if __name__ == "__main__":

    base_directory = '/network/scratch/l/lindongy/grid_blickets/conspec_ckpt/MultiDoorKeyEnv-6x6-2keys-v0-conspec-rec-PO-lr0.0006-intrinsR0.1-lrConSpec0.01-entropy0.02-num_mini_batch4-seed1'
    
    data_path = 'conspec_rollouts_epoch_3399.pth'
    cos_sim_path = 'cos_sim_epoch_3399.pth'
    frozen_sfbuffer_path = 'conspec_rollouts_frozen_epoch_3399.pth'
    buffer_path = 'buffer_epoch_3399.pth'
    
    data_full_path = os.path.join(base_directory, data_path)
    cos_full_path = os.path.join(base_directory, cos_sim_path)
    frozen_sfbuffer_full_path = os.path.join(base_directory, frozen_sfbuffer_path)
    buffer_full_path = os.path.join(base_directory, buffer_path)
    
    # Load buffer
    rollouts = torch.load(buffer_full_path)
    rollouts_numpy = {key: value.detach().cpu().numpy() for key, value in rollouts.items()}

    #==============================================================================

    # try rendering the observation
    t = 105
    i = 0  # index of rollout
    obs = rollouts_numpy['obs'][t, i] # shape: len_epi, num_rollouts, 3, 7, 7
    image_obs = render_obs(obs)
    plot_and_save_obs_image(image_obs, 'obs_image.png')

    episode_obs = rollouts_numpy['obs'][:, i]  # shape: len_epi, num_rollouts, 3, 7, 7
    save_episode_as_gif(episode_obs, filename='my_episode.gif', duration=200)


