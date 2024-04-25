import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# load trajectories
def load_trajectories(file_path):
    with open(file_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


def convert_obs_array_to_df(obs, reward_structure='baseline'):
    # Create a DataFrame from the numpy array with specified column names
    columns = ['A', 'B', 'C', 'D'] if reward_structure == 'baseline' else ['A', 'B', 'C', 'D', 'E']
    df = pd.DataFrame(obs, columns=columns)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and analyse trajectories')
    parser.add_argument('--file_path', type=str, default=None, help='absolute path to trajectories pkl file')
    args = parser.parse_args()
    print(args)
    
    trajectories = load_trajectories(args.file_path)

    # analyze components
    all_gts = [traj['gt'] for traj in trajectories]  # each element: str
    all_observations = [traj['observations'] for traj in trajectories]  # each element: epi_length x obs_dim
    all_actions = [traj['actions'] for traj in trajectories]  # each element: epi_length x action_dim
    all_rewards = [traj['rewards'] for traj in trajectories]  # each element: epi_length x 1
    all_terminals = [traj['terminals'] for traj in trajectories]  # each element: epi_length x 1

    # Group observations by different GTs (ground truths) and save to CSV
    unique_gts = set(all_gts)
    for gt in unique_gts:
        # Stack observations for current GT
        obs_for_gt = np.vstack([traj['observations'] for traj in trajectories if traj['gt'] == gt])
        df = convert_obs_array_to_df(obs_for_gt)
        
        # Save DataFrame to CSV
        filename = args.file_path.replace("trajectories.pkl", f"{gt}.csv")
        df.to_csv(filename, index_label='Index')