import numpy as np
import matplotlib.pyplot as plt


class GridBanditEnv:
    def __init__(self, grid_size=8, search_horizon=25, seed=None):
        self.grid_size = grid_size
        self.search_horizon = search_horizon
        np.random.seed(seed)
        self.normalized_reward_grid = self.create_spatially_correlated_rewards()
        self.visited_grid = np.zeros((grid_size, grid_size))
        self.reward_grid = np.zeros((grid_size, grid_size))  # unique for each round
        self.trials_played = 0
        self.grid_max = np.random.uniform(30, 40)  # changes every round


    def rbf_kernel(self, loc1, loc2, lambda_val):
        """Compute RBF kernel value for two locations."""
        distance = np.linalg.norm(np.array(loc1) - np.array(loc2))
        return np.exp(-distance / (2 * lambda_val**2))

    def create_spatially_correlated_rewards(self, lambda_val=2, num_centers=5):
        # Number of centers of influence
        center_locs = [tuple(np.random.randint(0, self.grid_size, 2)) for _ in range(num_centers)]

        # Assign random raw rewards to these centers
        center_rewards = np.random.rand(num_centers)

        # Create a grid for correlated rewards
        correlated_rewards = np.zeros((self.grid_size, self.grid_size))

        # Compute the influence of these centers on the entire grid using the RBF kernel
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for center, reward in zip(center_locs, center_rewards):
                    correlated_rewards[i, j] += reward * self.rbf_kernel((i, j), center, lambda_val)
        # print(correlated_rewards)

        # Normalize the rewards to lie between 0 and 40 and round to nearest integer
        min_reward = correlated_rewards.min()
        max_reward = correlated_rewards.max()
        normalized_rewards = (correlated_rewards - min_reward) / (max_reward - min_reward)
        # print(normalized_rewards)
        return normalized_rewards

    def plot_reward_heatmap(self, normalized=True):
        plt.figure(figsize=(10, 10))
        if normalized:
          reward_grid = self.normalized_reward_grid
        else:
          reward_grid = self.reward_grid
        # Plotting the heatmap
        plt.imshow(reward_grid, cmap='viridis', interpolation='none')

        # Adding the values of each cell to the heatmap
        for i in range(reward_grid.shape[0]):
            for j in range(reward_grid.shape[1]):
                plt.text(j, i, f"{reward_grid[i, j]:.2f}", ha='center', va='center', color='white')

        plt.colorbar()
        plt.title("Reward Grid Heatmap")
        plt.show()

    def plot_visited_reward_heatmap(self, n_trial, normalized=False):
        plt.figure(figsize=(10, 10))

        # Mask the reward grid based on visited tiles
        if normalized:
          reward_grid = self.normalized_reward_grid
        else:
          reward_grid = self.reward_grid
        masked_rewards = np.where(self.visited_grid, reward_grid, np.nan)

        # Plotting the heatmap
        plt.imshow(masked_rewards, cmap='viridis', interpolation='none', vmin=np.min(reward_grid), vmax=np.max(reward_grid))

        # Adding the values of each visited cell to the heatmap
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.visited_grid[i, j] == 1:
                    plt.text(j, i, f"{reward_grid[i, j]:.2f}", ha='center', va='center', color='white')

        plt.colorbar()
        plt.title(f"Visited Reward Grid Heatmap after trial {n_trial}")
        plt.show()

    def step(self, tile_coord, render=False):
        if self.trials_played >= self.search_horizon:
            raise Exception("End of search horizon, please reset the environment.")

        reward = np.random.normal(loc=self.reward_grid[tile_coord])  # corrupted by a gaussian noise N(0,1)
        norm_reward = self.normalized_reward_grid[tile_coord]
        self.visited_grid[tile_coord] = 1
        self.trials_played += 1
        if render:
          self.plot_visited_reward_heatmap(self.trials_played, normalized=False)
        return norm_reward, reward

    def reset(self):
        self.trials_played = 0
        self.visited_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Create spatially correlated rewards
        self.normalized_reward_grid = self.create_spatially_correlated_rewards()

        # Rescale the reward to between 5 and 45
        self.grid_max = np.random.uniform(30, 40)
        self.reward_grid = self.grid_max * self.normalized_reward_grid
        self.reward_grid += 5  # to avoid having negative reward given the noise

        # Randomly reveal a tile at the beginning
        self.current_tile = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        return self.current_tile
