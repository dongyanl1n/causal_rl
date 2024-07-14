import torch
import os 
import numpy as np
import matplotlib.pyplot as plt
from analysis_util import plot_cos_scores, render_obs, plot_and_save_obs_image, save_episode_as_gif, plot_high_cos_obs


def reshape_buffer(tensor, split_idx=16, len_epi=432):
    shape = tensor.shape
    if 2*split_idx in shape:
        return tensor
    elif shape[0] == len_epi*2*split_idx:
        return tensor.reshape(len_epi, 2*split_idx, *shape[1:])
    else:
        print(f"Warning: Unexpected shape {shape} for buffer tensor")
        return tensor
    
if __name__ == "__main__":

    base_directory = '/network/scratch/l/lindongy/grid_blickets/conspec_ckpt/MultiDoorKeyEnv-6x6-2keys-v0-conspec-rec-PO-lr0.0006-intrinsR0.2-lrConSpec0.006-entropy0.02-num_mini_batch4-seed5'
    
    data_path = 'conspec_rollouts_epoch_5600.pth'
    cos_sim_path = 'cos_sim_epoch_5600.pth'
    frozen_sfbuffer_path = 'conspec_rollouts_frozen_epoch_5600.pth'
    buffer_path = 'buffer_epoch_5600.pth'
    
    data_full_path = os.path.join(base_directory, data_path)
    cos_full_path = os.path.join(base_directory, cos_sim_path)
    frozen_sfbuffer_full_path = os.path.join(base_directory, frozen_sfbuffer_path)
    buffer_full_path = os.path.join(base_directory, buffer_path)
    
    split_idx = 16  # 16 successes
    
    # Load and split conspec_rollouts
    conspec_rollouts = torch.load(data_full_path)
    sf_buffer = {}
    for key in conspec_rollouts.keys():
        data = conspec_rollouts[key]
        reshaped_data = reshape_buffer(data)
        assert reshaped_data.shape[1] == 2*split_idx
        sf_buffer[f'{key}_S'] = reshaped_data[:, :split_idx].detach().cpu().numpy()
        sf_buffer[f'{key}_F'] = reshaped_data[:, split_idx:].detach().cpu().numpy()

    # Load and split conspec_rollouts_frozen
    conspec_rollouts_frozen = torch.load(frozen_sfbuffer_full_path)
    frozen_sf_buffers = {}
    for i in conspec_rollouts_frozen.keys():
        frozen_sf_buffers[i] = {}
        for key in conspec_rollouts_frozen[i].keys():
            data = conspec_rollouts_frozen[i][key]
            reshaped_data = reshape_buffer(data)
            assert reshaped_data.shape[1] == 2*split_idx
            frozen_sf_buffers[i][f'{key}_S'] = reshaped_data[:, :split_idx].detach().cpu().numpy()
            frozen_sf_buffers[i][f'{key}_F'] = reshaped_data[:, split_idx:].detach().cpu().numpy()
    
    # Load buffer
    rollouts = torch.load(buffer_full_path)
    rollouts_numpy = {key: value.detach().cpu().numpy() for key, value in rollouts.items()}

    # Load cos_sim
    cos_checkpoint = torch.load(cos_full_path)
    cos_max_scores = cos_checkpoint['cos_max_scores'].detach().cpu().numpy()
    cos_max_indices = cos_checkpoint['max_indices'].detach().cpu().numpy()
    cos_scores = cos_checkpoint['cos_scores'].detach().cpu().numpy()
    cos_max_scoresS = cos_max_scores[:split_idx]
    cos_max_scoresF = cos_max_scores[split_idx:]
    cos_max_indicesS = cos_max_indices[:split_idx]
    cos_max_indicesF = cos_max_indices[split_idx:]
    cos_scoresS = cos_scores[:, :split_idx]
    cos_scoresF = cos_scores[:, split_idx:]

    sf_buffer_obs = np.concatenate((sf_buffer['obs_S'], sf_buffer['obs_F']), axis=1)
    num_prototypes = cos_max_scores.shape[1]

    obs_batchS = np.array([frozen_sf_buffers[i]['obs_S'] for i in range(num_prototypes)]).transpose(1, 2, 0, 3, 4, 5)
    obs_batchF = np.array([frozen_sf_buffers[i]['obs_F'] for i in range(num_prototypes)]).transpose(1, 2, 0, 3, 4, 5)
    assert obs_batchS.shape[0:3] == cos_scoresS.shape
    assert obs_batchF.shape[0:3] == cos_scoresF.shape


    figure_path = os.path.join(base_directory, "figures")
    S_figure_path = os.path.join(figure_path, "success")
    F_figure_path = os.path.join(figure_path, "fail")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path, exist_ok=True)
    if not os.path.exists(S_figure_path):
        os.makedirs(S_figure_path, exist_ok=True)
    if not os.path.exists(F_figure_path):
        os.makedirs(F_figure_path, exist_ok=True)

    #==============================================================================

    # Plot cos scores
    # plot_cos_scores(cos_scoresS, S_figure_path)
    # plot_cos_scores(cos_scoresF, F_figure_path)
    
    # try rendering the observation
    # t = 105
    i = 0  # index of rollout
    # obs = rollouts_numpy['obs'][t, i] # shape: len_epi, num_rollouts, 3, 7, 7
    # image_obs = render_obs(obs)
    # plot_and_save_obs_image(image_obs, 'obs_image.png')

    # episode_obs = rollouts_numpy['obs'][:, i]  # shape: len_epi, num_rollouts, 3, 7, 7
    # save_episode_as_gif(episode_obs, filename='my_episode.gif', duration=200)

    plot_high_cos_obs(cos_scoresS, obs_batchS, threshold=0.95, save_path=S_figure_path)
    plot_high_cos_obs(cos_scoresF, obs_batchF, threshold=0.95, save_path=F_figure_path)

