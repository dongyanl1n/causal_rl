import gym
from gym import spaces
import numpy as np
from tqdm import tqdm
import pandas as pd

class CausalKeyToDoorEnv(gym.Env):
    def __init__(self, env_config) -> None:
        """
        fully observable MDP version of key to door task.
        State space: multibinary of size N, where N = number of apples + 3 (for key, door, and reward)
        Action space: Discrete(2) (act on it or ignore)
        """
        super(CausalKeyToDoorEnv, self).__init__()

        self.n_apples = env_config.get("n_apples", 3)
        self.add_apple_reward = env_config.get("add_apple_reward", True)
        self.apple_reward = env_config.get("apple_reward", 1)
        self.add_final_reward = env_config.get("add_final_reward", True)
        self.final_reward = env_config.get("final_reward", 100)


        # Define observation space
        self.n_states = self.n_apples + 3
        self.observation_space = spaces.MultiBinary(self.n_states)  # [key, apple1, apple2, apple3, door, reward]

        # Define action space
        self.action_space = spaces.Discrete(2)  # 1 = act on it, 0 = ignore

        # Initialize other variables and state

    def reset(self):
        # Reset the environment to its initial state
        self.done = False
        self.state = np.zeros(self.n_states)
        self.object = 'key'  # key, apple, door, reward
        self.apple_counter = 0  # count of apples seen so far
        self.has_key = False
        self.to_reward = None
        return self.state



    def step(self, action: int):
        # Perform one step in the environment based on the given action
        # Return the next state, reward, done, and info
        # At the start, if agent chooses to act on the key, it will get the key
        if self.object == 'key':
            assert np.all(self.state == np.zeros(self.n_states))
            self.state[0] = action
            self.has_key = True if action == 1 else False
            self.object = 'apple'
            reward = 0 

        # If agent chooses to act on an apple, it will get the apple
        elif self.object == 'apple':
            self.state[self.apple_counter + 1] = action
            self.apple_counter += 1
            if action == 1:
                if self.add_apple_reward:
                    reward = self.apple_reward
            else:
                reward = 0
            if self.apple_counter == self.n_apples:
                self.object = 'door'
            else:
                self.object = 'apple'

        # If agent chooses to act on the door when it has the key, it will unlock the door
        elif self.object == 'door':
            assert np.all(self.state[-2:] == np.zeros(2))
            if self.has_key:
                self.state[-2] = action  # unlock door if action is 1
            else:
                self.state[-2] = 0 # door remains locked
            self.object = 'reward'
            reward = 0

        # If agent chooses to act on the reward (only possible after unlocking the door), it will get the reward
        elif self.object == 'reward':
            assert self.state[-1] == 0
            self.to_reward = False
            if self.state[-2] == 1:  # door unlocked
                self.state[-1] = action
                if action == 1:
                    self.to_reward = True
                # self.done = True
                # if self.add_final_reward:
                #     reward = self.final_reward
                # else:
                #     reward = 0
            else:
                self.state[-1] = 0
                # self.done = True
                # reward = 0
            self.object = 'terminate'
            reward = 0
        
        elif self.object == 'terminate':
            if self.add_final_reward and self.to_reward:
                reward = self.final_reward
            else:
                reward = 0
            self.done = True
        

        observation = self.state
        return observation, reward, self.done, {}
            


def test_env():
    n_apples = 6
    n_epi = 1000
    env = CausalKeyToDoorEnv({
        "n_apples": n_apples,
        "add_apple_reward": True,
        "apple_reward": 1,
        "add_final_reward": True,
        "final_reward": 100
    })
    observs = []
    for i_epi in range(n_epi):
        obs = env.reset()
        # test environment with random action
        while not env.done:
            print(f"State: {obs}")
            observs.append(obs)
            print(f"Object: {env.object}")
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Done: {done}")
            obs = next_obs
    observs = np.array(observs)
    # convert observs to int
    observs = observs.astype(int)
    
    # Define column headers
    column_headers = ['K', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'D', 'R']
    
    # Save observs as CSV file with headers
    df = pd.DataFrame(observs, columns=column_headers)
    df.to_csv(f"ktd_{n_apples}apples_{n_epi}epi.csv", index=False)


if __name__ == "__main__":
    test_env()

    