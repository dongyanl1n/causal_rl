import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

class GPR_UCB_Agent:
    def __init__(self, grid_size=8, lambda_val=2.0, beta=2.0, tau=1.0, seed=None):
        self.grid_size = grid_size
        self.lambda_val = lambda_val
        self.beta = beta
        self.tau = tau
        np.random.seed(seed)
        self.X_train = []  # List of previously chosen tiles
        self.y_train = []  # List of rewards received for chosen tiles

    def rbf_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(np.array(x1) - np.array(x2))**2 / (2 * self.lambda_val**2))

    def predict(self, X):
        if len(self.X_train) == 0:
            return np.zeros(len(X)), np.zeros(len(X))

        K = np.zeros((len(self.X_train), len(self.X_train)))
        for i in range(len(self.X_train)):
            for j in range(len(self.X_train)):
                K[i, j] = self.rbf_kernel(self.X_train[i], self.X_train[j])

        K_s = np.zeros((len(self.X_train), len(X)))
        for i in range(len(self.X_train)):
            for j in range(len(X)):
                K_s[i, j] = self.rbf_kernel(self.X_train[i], X[j])

        K_ss = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                K_ss[i, j] = self.rbf_kernel(X[i], X[j])

        K_inv = np.linalg.inv(K + 1 * np.eye(len(self.X_train))) # std of reward noise is 1

        # Posterior mean
        mu_s = K_s.T.dot(K_inv).dot(self.y_train)

        # Posterior covariance
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        # Ensuring diagonal entries are non-negative
        var_s = np.diag(cov_s)
        var_s = np.copy(var_s)
        var_s[var_s < 0] = 0

        return mu_s, var_s

    def choose_tile(self):
        all_tiles = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        mu, var = self.predict(all_tiles)
        q_values = mu + self.beta * np.sqrt(var)

        # Convert q-values to choice probabilities via softmax
        choice_probs = softmax(q_values / self.tau)
        if np.any(np.isnan(choice_probs)):
          breakpoint()
        chosen_tile_index = np.random.choice(len(all_tiles), p=choice_probs)
        return all_tiles[chosen_tile_index]

    def update(self, chosen_tile, reward):
        self.X_train.append(chosen_tile)
        self.y_train.append(reward)

    def reset(self):
        self.X_train = []
        self.y_train = []
