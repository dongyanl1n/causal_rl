import numpy as np
import matplotlib.pyplot as plt
from envs.GridBandit import GridBanditEnv
from models.q_learning import QLearningAgent
from models.sr_agent import SRAgent
from models.gp_ucb import GPR_UCB_Agent
import matplotlib.cm as cm
from tqdm import tqdm
import scipy.stats as stats

def count_unique_tiles(tile_coordinates):
    """
    Count the number of unique tile coordinates chosen for each trial in each round.

    Parameters:
    - tile_coordinates: A list of lists where each inner list consists of the chosen tile coordinates for each trial.

    Returns:
    - A list of counts of unique tile coordinates for each trial in each round.
    """
    unique_tile_counts = []

    for round_coords in tile_coordinates:
        # Using set to identify unique tile coordinates for the round
        round_set = set(tuple(coord) for coord in round_coords)
        unique_tile_counts.append(len(round_set))

    return unique_tile_counts

def calculate_tile_distances(tile_coordinates):
    """
    Calculate the distance between consecutive tiles in each round.

    Parameters:
    - tile_coordinates: A list of lists where each inner list consists of the chosen tile coordinates for each trial.

    Returns:
    - A list of lists where each inner list consists of distances between consecutive tiles.
    """
    distances = []
    for round_coords in tile_coordinates:
        round_distances = []
        for i in range(1, len(round_coords)):
            dist = np.linalg.norm(np.array(round_coords[i]) - np.array(round_coords[i - 1]))
            round_distances.append(dist)
        distances.append(round_distances)
    return distances


def plot_mean_reward_dict(norm_rewards_dict, colors, **kwargs):
    """
    Plots the normalized mean reward over trials, averaged across all rounds.

    Parameters:
    - rewards: A list of lists that consists of the rewards received for each trial of each round.
    """

    # Plot the results
    plt.figure(figsize=(10, 10))
    for i, (config, norm_rewards) in enumerate(norm_rewards_dict.items()):
      # Convert the list of lists to a numpy array for easier calculations
      normalized_rewards_array = np.array(norm_rewards)
      # Calculate the mean reward for each trial across all rounds
      mean_rewards = np.mean(normalized_rewards_array, axis=0)
      std_rewards = np.std(normalized_rewards_array, axis=0)
      plt.plot(mean_rewards, '-o', label=config, color=colors[i])
      plt.fill_between(np.arange(len(mean_rewards)), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.1, color=colors[i])
    plt.xlabel('Trial Number')
    plt.ylabel('Average Normalized Reward')
    plt.title('Mean Reward over Trials')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.show()

def plot_max_reward_dict(norm_rewards_dict, colors, **kwargs):
    """
    Plots the maximum reward earned up until a given trial, averaged across all rounds.

    Parameters:
    - rewards: A list of lists that consists of the rewards received for each trial of each round.
    """
    # Plot the results
    plt.figure(figsize=(10, 10))
    for i, (config, norm_rewards) in enumerate(norm_rewards_dict.items()):
      # Convert the list of lists to a numpy array for easier calculations
      normalized_rewards_array = np.array(norm_rewards)
      # Calculate the maximum reward earned up until the give trial
      normalized_maximum_cumulative_rewards = np.maximum.accumulate(normalized_rewards_array, axis=1)
      # average across rounds
      mean_rewards = np.mean(normalized_maximum_cumulative_rewards, axis=0)
      std_rewards = np.std(normalized_maximum_cumulative_rewards, axis=0)
      plt.plot(mean_rewards, '-o', label=config, color=colors[i])
      plt.fill_between(np.arange(len(mean_rewards)), mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.1, color=colors[i])
    plt.xlabel('Trial Number')
    plt.ylabel('Running Maximum Normalized Reward')
    plt.title('Max Reward over Trials')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.show()

def visualize_unique_tiles_dict(tile_coordinates_dict, colors, **kwargs):
    # Plotting the unique tile counts
    plt.figure(figsize=(10, 10))
    for i, (config, tile_coordinates) in enumerate(tile_coordinates_dict.items()):
      unique_tile_counts = count_unique_tiles(tile_coordinates)
      plt.plot(range(len(unique_tile_counts)), unique_tile_counts, label=config, color=colors[i], alpha=0.5)
    plt.xlabel('Round Number')
    plt.ylabel('Number of Unique Tiles')
    plt.title('Number of Unique Tiles Chosen per Round')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.show()

def visualize_relative_performance_dict(relative_performance_dict, colors, **kwargs):
    # Plotting the unique tile counts
    plt.figure(figsize=(10, 10))
    for i, (config, relative_performance) in enumerate(relative_performance_dict.items()):
      plt.plot(range(len(relative_performance)), relative_performance, label=config, color=colors[i], alpha=0.5)
    plt.xlabel('Round Number')
    plt.ylabel('Relative Performance')
    plt.ylim(0,1)
    plt.title('Relative Performance per Round')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.show()

def plot_distance_vs_reward_dict(tile_coordinates_dict, rewards_dict, colors, **kwargs):
    plt.figure(figsize=(10, 10))
    for i_config, (config, tile_coordinates, rewards) in enumerate(zip(tile_coordinates_dict.keys(),
                                                  tile_coordinates_dict.values(),
                                                  rewards_dict.values())):
        distances = calculate_tile_distances(tile_coordinates)
        # Flatten lists and associate each distance with the previous reward
        all_distances = []
        all_rewards = []
        for i in range(len(rewards)):  # rounds
          all_rewards.append(rewards[i][:-1])
          all_distances.append(distances[i])
          assert len(all_rewards)==len(all_distances), "all_rewards and all_distances must have the same length!"
        plt.scatter(all_rewards, all_distances, alpha=0.3, label=config, color=colors[i_config])
        # Run a linear regression between all_rewards and all_distances and plot the line
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(all_rewards).flatten(), np.array(all_distances).flatten())
        plt.plot(np.array(all_rewards).flatten(), intercept + slope*np.array(all_rewards).flatten(), 'r', color=colors[i_config])

    # TODO: add "fixed effect of a hierarchical Bayesian regression" as red baseline

    plt.xlabel('Previous Reward')
    plt.ylabel('Distance to Next Tile')
    plt.title('Distance to Next Tile vs. Previous Reward')
    plt.legend()
    # plt.tight_layout()
    plt.show()


# ================================================================
# run (multiple agents)
# ================================================================
def evaluate_agent_on_env(agent, env, num_rounds=100):
    """Evaluate agent's performance over multiple rounds."""
    if isinstance(agent, GPR_UCB_Agent):
        agent_config = (agent.lambda_val, agent.beta, agent.tau)  # for GPR-UCB
    elif isinstance(agent, SRAgent):
        agent_config = (agent.gamma, agent.gamma, agent.epsilon)
    elif isinstance(agent, QLearningAgent):
        agent_config = (agent.alpha, agent.gamma, agent.epsilon)
    else:
        raise NotImplementedError
    scores = []  # relative performance
    behaviour_data_dict = {
    "tile_coord": {
        agent_config: []
    },
    "reward": {
        agent_config: []
    },
    "norm_reward": {
        agent_config: []
    },
    "env_max_reward": {
        agent_config: []
    }  # maximum expected reward, pre-noise
    }

    for round in tqdm(range(num_rounds)):
        env.reset()
        agent.reset()
        behaviour_data_dict["env_max_reward"][agent_config].append(np.max(env.reward_grid))
        total_reward = 0
        total_normalized_reward = 0
        tile_coord_round = []
        reward_round = []
        norm_reward_round = []
        for t in range(env.search_horizon):

            if isinstance(agent, SRAgent):  # For SR agent
                chosen_tile = agent.choose_tile(np.argwhere(env.visited_grid == 1))
                normalized_reward, reward = env.step(chosen_tile, render=False)
                agent.update(chosen_tile, reward)

            elif isinstance(agent, QLearningAgent):  # For Q-Learning agent
                chosen_tile = agent.choose_action(env.visited_grid)
                normalized_reward, reward = env.step(chosen_tile, render=False)
                agent.update(chosen_tile, reward, env.visited_grid)

            elif isinstance(agent, GPR_UCB_Agent):  # For GP-UCB agent
                chosen_tile = agent.choose_tile()
                normalized_reward, reward = env.step(chosen_tile, render=False)
                agent.update(chosen_tile, reward)

            else:
                raise NotImplementedError

            total_reward += reward
            total_normalized_reward += normalized_reward
            reward_round.append(reward)
            norm_reward_round.append(normalized_reward)
            tile_coord_round.append(chosen_tile)
        scores.append(total_normalized_reward/env.search_horizon)
        behaviour_data_dict["tile_coord"][agent_config].append(tile_coord_round)
        behaviour_data_dict["reward"][agent_config].append(reward_round)
        behaviour_data_dict["norm_reward"][agent_config].append(norm_reward_round)
    return scores, behaviour_data_dict




def main_GPR():
    # Run GPR-UCB agent with different parameter settings on GridBandit
    # Values we want to experiment with
    lambda_vals = [1.0, 2.0, 3.0, 4.0]
    beta_vals = [1.0, 2.0, 3.0]
    tau_vals = [0.5, 1.0, 1.5]

    # Store results
    results = {}  # relative peformance
    behaviour_data_all = {
        "tile_coord": {},
        "reward": {},
        "norm_reward": {},
        "env_max_reward": {}
    }

    # Create environment instance
    env = GridBanditEnv(seed=1)

    # Iterate over all combinations
    for lambda_val in lambda_vals:
        for beta_val in beta_vals:
            for tau_val in tau_vals:
                print(f"Lambda: {lambda_val}, Beta: {beta_val}, Tau: {tau_val}")
                agent = GPR_UCB_Agent(beta=beta_val, tau=tau_val, lambda_val=lambda_val)
                scores, behaviour_data = evaluate_agent_on_env(agent, env, num_rounds=50)
                config = (lambda_val, beta_val, tau_val)
                results[config] = scores
                for key in behaviour_data_all.keys():
                    if config not in behaviour_data_all[key].keys():
                        behaviour_data_all[key][config] = []
                    behaviour_data_all[key][config].extend(behaviour_data[key][config])
                print(f"Avg Score: {np.mean(scores)}")

    # colormap = cm.get_cmap("gist_ncar", len(results))
    colormap = cm.get_cmap("tab10", len(results))
    colors = [colormap(i) for i in range(len(results))]

    visualize_relative_performance_dict(results, colors=colors)
    plot_mean_reward_dict(behaviour_data_all["norm_reward"], colors=colors)
    plot_max_reward_dict(behaviour_data_all["norm_reward"], colors=colors)
    visualize_unique_tiles_dict(behaviour_data_all["tile_coord"], colors=colors)
    plot_distance_vs_reward_dict(behaviour_data_all["tile_coord"], behaviour_data_all["norm_reward"], colors=colors)

def main_SR():
    alpha_vals = [0.1, 0.2, 0.3, 0.4]
    gamma_vals = [0.9, 0.95, 0.99]
    epsilon_vals = [0.1, 0.2, 0.3, 0.4]

    # Store results
    results = {}  # relative peformance
    behaviour_data_all = {
        "tile_coord": {},
        "reward": {},
        "norm_reward": {},
        "env_max_reward": {}
    }

    # Create environment instance
    env = GridBanditEnv(seed=1)

    # Iterate over all combinations
    for alpha_val in alpha_vals:
        for gamma_val in gamma_vals:
            for epsilon_val in epsilon_vals:
                print(f"Alpha: {alpha_val}, Gamma: {gamma_val}, Epsilon: {epsilon_val}")
                agent = SRAgent(alpha=alpha_val, discount_factor=gamma_val, epsilon=epsilon_val)
                scores, behaviour_data = evaluate_agent_on_env(agent, env, num_rounds=50)
                config = (alpha_val, gamma_val, epsilon_val)
                results[config] = scores
                for key in behaviour_data_all.keys():
                    if config not in behaviour_data_all[key].keys():
                        behaviour_data_all[key][config] = []
                    behaviour_data_all[key][config].extend(behaviour_data[key][config])
                print(f"Avg Score: {np.mean(scores)}")

    # colormap = cm.get_cmap("gist_ncar", len(results))
    colormap = cm.get_cmap("tab10", len(results))
    colors = [colormap(i) for i in range(len(results))]

    visualize_relative_performance_dict(results, colors=colors)
    plot_mean_reward_dict(behaviour_data_all["norm_reward"], colors=colors)
    plot_max_reward_dict(behaviour_data_all["norm_reward"], colors=colors)
    visualize_unique_tiles_dict(behaviour_data_all["tile_coord"], colors=colors)
    plot_distance_vs_reward_dict(behaviour_data_all["tile_coord"], behaviour_data_all["norm_reward"], colors=colors)


if __name__ == "__main__":
    main_SR()