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
    argsdict = args.__dict__
    print(argsdict)
    
    trajectories = load_trajectories(args.file_path)

    if "trajectories.pkl" in args.file_path or "v2.pkl" in args.file_path:  # collected from RL agent or random policy

        # analyze components
        all_gts = [traj['gt'] for traj in trajectories]  # each element: str
        all_observations = [traj['observations'] for traj in trajectories]  # each element: epi_length x obs_dim
        all_actions = [traj['actions'] for traj in trajectories]  # each element: epi_length x action_dim
        all_rewards = [traj['rewards'] for traj in trajectories]  # each element: epi_length x 1
        all_terminals = [traj['terminals'] for traj in trajectories]  # each element: epi_length x 1

        # calculate mean return
        all_returns = [np.sum(traj['rewards']) for traj in trajectories]
        mean_return = np.mean(all_returns)
        print(f"Mean return: {mean_return}")

        # separate returns by gts
        for gt in set(all_gts):
            returns_for_gt = [np.sum(traj['rewards']) for traj in trajectories if traj['gt'] == gt]
            print(f"GT: {gt}")
            print(f"Mean return: {np.mean(returns_for_gt)}")
            actions_for_gt = [traj['actions'] for traj in trajectories if traj['gt'] == gt]
            # Check if all actions are the same across trajectories for each gt
            actions_consistent = np.all(np.array([np.array_equal(actions_for_gt[0], act) for act in actions_for_gt]))
            print(f"Actions consistent across trajectories for GT {gt}: {actions_consistent}")
            # if all actions are the same, print what action it is
            if actions_consistent:
                print(f"Actions: {actions_for_gt[0]}")
            num_exploration_steps = []
            for act in actions_for_gt:  # Assuming each act is a array of shape [epi_length, action_dim]
                # count where the first 1 appears in the last dimension of action
                idx_first_1 = np.where(act[:, -1] == 1)[0][0]
                num_exploration_steps.append(idx_first_1)
            mean_exploration_steps = np.mean(num_exploration_steps)
            print(f"Mean number of exploration steps: {mean_exploration_steps}")


        # Group observations by different GTs (ground truths) and save to CSV <--- for GFlowNet
        # unique_gts = set(all_gts)
        # for gt in unique_gts:
        #     # Stack observations for current GT
        #     obs_for_gt = np.vstack([traj['observations'] for traj in trajectories if traj['gt'] == gt])
        #     df = convert_obs_array_to_df(obs_for_gt)
            
        #     # Save DataFrame to CSV
        #     filename = args.file_path.replace("trajectories.pkl", f"{gt}.csv")
        #     df.to_csv(filename, index_label='Index')
    
    elif "iteration" in args.file_path:  # collected from decision transformer
        # analyze components. length of each list is equal to the num_eval_traj inmodels/decision-transformer/experiment.py
        all_gts = [traj['gt'] for traj in trajectories]
        all_observations = [traj['observations'].cpu().numpy() for traj in trajectories]
        all_actions = [traj['actions'].cpu().numpy() for traj in trajectories]
        all_rewards = [traj['rewards'].cpu().numpy() for traj in trajectories]
        all_target_returns = [traj['target_returns'].cpu().numpy() for traj in trajectories]
        all_returns = [traj['returns'] for traj in trajectories]
        all_lengths = [traj['length'] for traj in trajectories]

        print(f"Mean return: {np.mean(all_returns)}")
        print(f"Mean length: {np.mean(all_lengths)}")

        # Separate returns by gts
        for gt in set(all_gts):
            returns_for_gt = np.array([traj['returns'] for traj in trajectories if traj['gt'] == gt])
            print(f"GT: {gt}, mean return: {np.mean(returns_for_gt)}")
            actions_for_gt = [traj['actions'].cpu().numpy() for traj in trajectories if traj['gt'] == gt]

            # Check if all actions are the same across trajectories for each gt
            actions_consistent = np.all(np.array([np.array_equal(actions_for_gt[0], act) for act in actions_for_gt]))
            print(f"Actions consistent across trajectories for GT {gt}: {actions_consistent}")
            # if all actions are the same, print what action it is
            if actions_consistent:
                print(f"Actions: {actions_for_gt[0]}")

            num_exploration_steps = []
            for act in actions_for_gt:  # Assuming each act is a array of shape [epi_length, action_dim]
                # count where the first 1 appears in the last dimension of action
                idx_first_1 = np.where(act[:, -1] == 1)[0][0]
                num_exploration_steps.append(idx_first_1)
            mean_exploration_steps = np.mean(num_exploration_steps)
            print(f"Mean number of exploration steps: {mean_exploration_steps}")