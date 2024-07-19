import torch
import numpy as np
import os
from Conspec.prototype import prototypes
from Conspec.modelConSpec import EncoderConSpec
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cos_max_scores_heatmap(cos_max_scores_all):
    # for each rollout, plot the heatmap of cos_max_scores
    for i in range(cos_max_scores_all.shape[0]):  # for each rollout
        sns.heatmap(cos_max_scores_all[i].detach().cpu().numpy())
        plt.title(f'Rollout {i}')
        plt.xlabel('Buffer Prototype Index')
        plt.ylabel('Prototype Index')
        plt.savefig(f'cos_max_scores/rollout_{i}.png')
        plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obsspace = Box(0, 255, (3, 7, 7))
actionspace = Discrete(7)

# define model
num_prototypes = 8
conspec_encoder = EncoderConSpec(
            obsspace.shape,
            actionspace.n,
            base_kwargs={'recurrent': True,  # only to ensure it has the same hidden size as RL model; forward pass through CNN actually does not include GRU
                     'observation_space': obsspace})  # envs.observation_space.shape,
conspec_encoder.to(device)
conspec_prototypes = prototypes(input_size=conspec_encoder.recurrent_hidden_state_size, hidden_size=1010, 
                            num_prototypes=num_prototypes, device=device)
conspec_prototypes.to(device)

# load weights
base_path = '/network/scratch/l/lindongy/MultiDoorKeyEnv-6x6-2keys-v0-conspec-rec-PO-11111111-lr0.0009-intrinsR0.1-lrConSpec0.007-entropy0.02-seed4'
ckpt_epi = 7299
checkpoint_path = os.path.join(base_path, f"model_checkpoint_epoch_{ckpt_epi}.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
conspec_encoder.load_state_dict(checkpoint['encoder_state_dict'])
conspec_prototypes.prototypes.load_state_dict(checkpoint['prototypes'])
conspec_prototypes.layers1.load_state_dict(checkpoint['layers1'])
conspec_prototypes.layers2.load_state_dict(checkpoint['layers2'])

# load obs data
# cos_sim_path = 'cos_sim_epoch_7299.pth'
frozen_sfbuffer_path = 'conspec_rollouts_frozen_epoch_7299.pth'
buffer_path = 'buffer_epoch_7299.pth'
data_path = 'conspec_rollouts_epoch_7299.pth'

# cos_full_path = os.path.join(base_path, 'data', cos_sim_path)
frozen_sfbuffer_full_path = os.path.join(base_path, 'data', frozen_sfbuffer_path)
buffer_full_path = os.path.join(base_path, 'data', buffer_path)
data_full_path = os.path.join(base_path, 'data', data_path)

split_idx = 16  # 16 successes
    
conspec_rollouts = torch.load(data_full_path)
conspec_rollouts_frozen = torch.load(frozen_sfbuffer_full_path)
rollouts = torch.load(buffer_full_path)

#================== calculate cosine scores on frozen buffers...================
# cos_scores_all = []
# cos_max_scores_all = []
# cos_max_indices_all = []
# embeddings_all = []

# for i_prototype in range(num_prototypes):  # loop through frozen SF buffers for each prototype
#     obs_batch = conspec_rollouts_frozen[i_prototype]['obs']
#     recurrent_hidden_states_batch = conspec_rollouts_frozen[i_prototype]['hidden_states']
#     masks_batch = conspec_rollouts_frozen[i_prototype]['masks']
#     actions_batch = conspec_rollouts_frozen[i_prototype]['actions']

#     hidden = conspec_encoder.retrieve_hiddens(obs_batch, recurrent_hidden_states_batch, masks_batch)
#     hidden = hidden.reshape(432, 32, -1)  # Assuming hidden is of shape (ep_length*num_rollouts, 512)
#     embeddings_all.append(conspec_prototypes.layers2[i_prototype](conspec_prototypes.layers1[i_prototype](hidden)))

#     cos_max_score, max_inds, _, cos_scores, _ = conspec_prototypes(
#         hidden, -1, loss_ortho_scale=0.1
#     )  # cos_score between SFbuffer_for_prototype_i and all prototypes -- here the i_prototype and loss_ortho_scale arguments don't matter!

#     cos_scores_all.append(cos_scores)
#     cos_max_scores_all.append(cos_max_score)
#     cos_max_indices_all.append(max_inds)

# # Convert lists to tensors if needed
# # last dimension is frozen buffer for each prototype
# cos_scores_all = torch.stack(cos_scores_all, dim=-1)  # ep_length, num_rollouts, num_prototypes, num_prototypes
# cos_max_scores_all = torch.stack(cos_max_scores_all, dim=-1)  # num_rollouts, num_prototypes, num_prototypes
# cos_max_indices_all = torch.stack(cos_max_indices_all, dim=-1)  # num_rollouts, num_prototypes, num_prototypes
# embeddings_all = torch.stack(embeddings_all, dim=-1)  # ep_length, num_rollouts, 1010, num_prototypes

# plot_cos_max_scores_heatmap(cos_max_scores_all)


# ========= look at observations in the current buffer that elicit highest cos score with each prototype ======
# obs_batch = rollouts['obs'][:-1].reshape(432*8, 3, 7, 7)
# recurrent_hidden_states_batch = rollouts['hidden_states'][:-1].reshape(432*8, -1)
# masks_batch = rollouts['masks'][:-1].reshape(432*8, 1)
# actions_batch = rollouts['actions'].reshape(432*8, 1)

# # Equivalent to self.encoder.retrieve_hiddens
# hidden = conspec_encoder.retrieve_hiddens(obs_batch, recurrent_hidden_states_batch, masks_batch)
# hidden = hidden.reshape(432, 8, -1)
# cos_max_score, max_inds, _, cos_scores, _ = conspec_prototypes(
#     hidden, -1, loss_ortho_scale=0.1
# )  # cos_score between rollouts and all prototypes -- here the i_prototype and loss_ortho_scale arguments don't matter!
# print(cos_scores.shape)  # (432, 8, 8)  # 432, num_rollouts, num_prototypes
# obs_batch = obs_batch.reshape(432, 8, 3, 7, 7)

# # for each prototype, identify the observations that elicit high cos score
# threshold = 0.999
# for i_prototype in range(num_prototypes):
#     high_cos_indices = torch.where(cos_scores[:, :, i_prototype] > threshold)
#     print(f'Prototype {i_prototype+1}: {len(high_cos_indices[0])} high cos score observations')
# # plot high cos score observations
# from analysis_util import plot_high_cos_obs
# plot_high_cos_obs(cos_scores, obs_batch, threshold=threshold, save_path='prototype_obs')

# ========= look at observations in the current buffer that elicit highest cos score with each prototype, and cluster their embeddings in 2D space ======
obs_batch = rollouts['obs'][:-1].reshape(432*8, 3, 7, 7)
recurrent_hidden_states_batch = rollouts['hidden_states'][:-1].reshape(432*8, -1)
masks_batch = rollouts['masks'][:-1].reshape(432*8, 1)
actions_batch = rollouts['actions'].reshape(432*8, 1)

# Equivalent to self.encoder.retrieve_hiddens
hidden = conspec_encoder.retrieve_hiddens(obs_batch, recurrent_hidden_states_batch, masks_batch)
hidden = hidden.reshape(432, 8, -1)
cos_max_score, max_inds, _, cos_scores, _ = conspec_prototypes(
    hidden, -1, loss_ortho_scale=0.1
)  # cos_score between rollouts and all prototypes -- here the i_prototype and loss_ortho_scale arguments don't matter!
print(cos_scores.shape)  # (432, 8, 8)  # 432, num_rollouts, num_prototypes
len_episode, num_rollouts, num_prototype = cos_scores.shape
cos_scores = cos_scores.reshape(len_episode*num_rollouts, num_prototype)
# obs_batch = obs_batch.reshape(len_episode, num_rollouts, 3, 7, 7)

# for each prototype, identify the observations that elicit high cos score
threshold = 0.99
high_cos_hidden_all = []
labels = []
for i_prototype in range(num_prototypes):
    high_cos_indices = torch.where(cos_scores[:, i_prototype] > threshold)
    print(f'Prototype {i_prototype+1}: {len(high_cos_indices[0])} high cos score observations')
    if len(high_cos_indices[0]) == 0:
        continue
    high_cos_obs = obs_batch[high_cos_indices[0]]
    high_cos_hidden_states = recurrent_hidden_states_batch[high_cos_indices[0]]
    high_cos_masks = masks_batch[high_cos_indices[0]]
    high_cos_hidden = conspec_encoder.retrieve_hiddens(high_cos_obs, high_cos_hidden_states, high_cos_masks)
    high_cos_hidden_all.append(high_cos_hidden)
    labels.extend([i_prototype] * len(high_cos_hidden))


# concatenate high_cos_hiddden_all, run dimensionality reduction, and label with labels
from sklearn.manifold import TSNE
if high_cos_hidden_all:
    all_hidden = torch.cat(high_cos_hidden_all, dim=0)

    # Convert to numpy for sklearn
    all_hidden_np = all_hidden.detach().cpu().numpy()
    labels_np = np.array(labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(all_hidden_np)

    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels_np, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of high cosine similarity hidden states')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.savefig('t-SNE.png')
else:
    print("No hidden states with high cosine similarity found.")


