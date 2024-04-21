import numpy as np

class QLearningAgent:
    def __init__(self, grid_size=8, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        self.grid_size = grid_size
        self.Q = np.zeros((grid_size, grid_size))  # Initializing Q-values to zero
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        np.random.seed(seed)

    def choose_action(self, visited_grid):
        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration: Randomly select an unvisited tile
            unvisited_tiles = np.argwhere(visited_grid == 0)
            return tuple(unvisited_tiles[np.random.choice(len(unvisited_tiles))])
        else:
            # Exploitation: Choose the tile with the highest Q-value among unvisited tiles
            max_val = -np.inf
            best_tile = None
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if visited_grid[i, j] == 0 and self.Q[i, j] > max_val:
                        max_val = self.Q[i, j]
                        best_tile = (i, j)
            return best_tile

    def update(self, tile_coord, reward, visited_grid):
        i, j = tile_coord
        # Update Q-value for the selected action
        max_future_q = np.max(self.Q[visited_grid == 0])
        self.Q[i, j] += self.alpha * (reward + self.gamma * max_future_q - self.Q[i, j])

    def reset(self):
        self.Q = np.zeros((self.grid_size, self.grid_size))  # Initializing Q-values to zero