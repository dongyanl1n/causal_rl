import numpy as np

class SRAgent:
    def __init__(self, grid_size=8, alpha=0.1, discount_factor=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.M = np.eye(grid_size * grid_size)  # Initialize the SR matrix as an identity matrix
        self.reward_estimates = np.zeros(grid_size * grid_size)

    def _flatten_coord(self, coord):
        return coord[0] * self.grid_size + coord[1]

    def _unflatten_index(self, index):
        return divmod(index, self.grid_size)

    def update(self, current_coord, reward):
        current_idx = self._flatten_coord(current_coord)

        # Update the SR using TD learning
        td_error = np.eye(self.grid_size * self.grid_size)[current_idx] + self.gamma * self.M[current_idx] - self.M[current_idx]
        self.M[current_idx] += self.alpha * td_error

        # Update the estimated reward for the current state
        self.reward_estimates[current_idx] = reward

    def choose_tile(self, visited_tiles):
        # Compute the value of each state using the SR
        values = np.dot(self.M, self.reward_estimates)

        # Set the value of visited tiles to a very negative value to avoid revisiting
        for tile in visited_tiles:
            idx = self._flatten_coord(tile)
            values[idx] = -1e9

        # Choose the state with the highest value
        best_tile_index = np.argmax(values)
        return self._unflatten_index(best_tile_index)

    def reset(self):
        self.M = np.eye(self.grid_size * self.grid_size)  # Initialize the SR matrix as an identity matrix
        self.reward_estimates = np.zeros(self.grid_size * self.grid_size)