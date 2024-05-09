import numpy as np
import torch
import argparse

# parse args: seed, hypothesis, random_action
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hypothesis', type=str, default='ABCconj')
parser.add_argument('--random_action', type=bool, default=False)  # TODO: fix this
args = parser.parse_args()
seed = args.seed
hypothesis = args.hypothesis
random_action = args.random_action
print(seed, hypothesis, random_action)

# Set seed for reproducibility
np.random.seed(seed)  # Set seed for numpy
torch.manual_seed(seed)  # Set seed for torch
torch.cuda.manual_seed(seed)  # Set seed for current CUDA device

# If you are using CUDA with PyTorch and want to make it deterministic:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = True
if torch.cuda.is_available() and use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from envs.causal_env_v0 import CausalEnv_v1, ABconj, ACconj, BCconj, Adisj, Bdisj, Cdisj, ABCdisj, ABCconj, ABdisj, ACdisj, BCdisj
from models.replay_buffer import ReplayBuffer
from models.core import Core
import pandas as pd
from pgmpy.estimators import MmhcEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from collections import defaultdict
from collections import deque
from tqdm import tqdm
import torch


def visualize_graph(model):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    pos = nx.layout.circular_layout(G)  # Positions for all nodes
    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    # Edges
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray')
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    plt.title('Bayesian Network Learned Structure')
    plt.show()



def one_hot_action_to_tuple(action, n_blickets):
    # here, action is a flattened vector of dimension n_blickets*2
    return (action % n_blickets, action // n_blickets)


# Define the RL agent
class Agent:
    def __init__(self, env, buffer_capacity=1000, batch_size=10, convergence_threshold=5, hidden_size=128, device=device, random_action=False):
        self.env = env
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.state_visit_counts = defaultdict(lambda: defaultdict(int))
        self.previous_graphs = deque(maxlen=convergence_threshold)
        self.convergence_threshold = convergence_threshold
        self.core = Core(env.observation_space.shape[0], env._n_blickets*2, hidden_size, device)
        self.hidden_size = hidden_size
        self.device = device
        self.random_action = random_action
        self.model = None

    def infer_causal_graph(self):
        data = pd.DataFrame([entry[0] for entry in self.replay_buffer.get_all_data()])  # Ensure data is in DataFrame
        if data.empty:
            return None
        mmhc_estimator = MmhcEstimator(data)
        structure = mmhc_estimator.estimate()
        model = BayesianNetwork(structure.edges())
        # Here we need to fit the CPDs using, for example, Maximum Likelihood Estimation
        model.fit(data, estimator=MaximumLikelihoodEstimator)
        self.model = model
        return self.model

    def select_action(self, state, random_action=False):
        if random_action:
            action = self.env.action_space.sample()
            self.state_visit_counts[tuple(state)][action] += 1
            return action
        else:
            action, value, log_prob = self.core.select_action(state)
            self.state_visit_counts[tuple(state)][action] += 1
            return action, value, log_prob

    def compute_intrinsic_reward(self, state, action, epsilon=0.001):
        visit_count = self.state_visit_counts[tuple(state)][action]
        intrinsic_reward = (self.env._steps / (visit_count + self.env._steps * epsilon))**0.5  # Simple count-based intrinsic reward
        return intrinsic_reward

    def has_converged(self):
        if len(self.previous_graphs) < self.convergence_threshold:
            return False
        return all(g == self.previous_graphs[0] for g in self.previous_graphs)

    def train(self, episodes):
        losses = []
        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            self.core.reset_hidden()
            done = False
            rewards, log_probs, values, dones = [], [], [], []
            while not done:
                if self.random_action:
                    action = self.select_action(state, random_action=self.random_action)
                    next_state, reward, done, _ = self.env.step(action)
                    intrinsic_reward = self.compute_intrinsic_reward(state, action)
                    total_reward = reward + intrinsic_reward
                    self.replay_buffer.push(state, action, total_reward, next_state, done)
                    state = next_state
                else:
                    action, value, log_prob = self.select_action(state)
                    env_action = one_hot_action_to_tuple(action, self.env._n_blickets)
                    next_state, reward, done, _ = self.env.step(env_action)
                    intrinsic_reward = self.compute_intrinsic_reward(state, action)
                    total_reward = reward + 0.001 * intrinsic_reward
                    self.replay_buffer.push(state, action, total_reward, next_state, done)
                    rewards.append(total_reward)
                    log_probs.append(log_prob)
                    values.append(value)
                    dones.append(done)
                    state = next_state

            if not self.random_action:
                loss = self.core.learn(log_probs, values, rewards, dones)
                losses.append(loss)

            # Periodically infer the causal graph
            if episode % 10 == 0:
                self.model = self.infer_causal_graph()
                if self.model:
                    edges = frozenset(self.model.edges())
                    self.previous_graphs.append(edges)
                    print("Learned model structure:", edges)
                    if self.has_converged():
                        print(f"Model has converged after {episode} episodes.")
                        break
                if not self.random_action:
                    # report running average loss
                    print(f"Episode {episode}. Running average loss: {sum(losses[-10:]) / 10:.2f}")


    def inference(self, query_vars, evidence):
        """
        Perform inference given evidence and return the probabilities for query variables.
        
        :param query_vars: List of variable indices for which to compute probabilities
        :param evidence: Dictionary of observed variables and their values
        :return: A dictionary of probability distributions for each query variable
        """
        assert self.model is not None, "Model has not been learned yet."
        assert self.has_converged(), "Model has not converged yet."
        assert isinstance(self.model, BayesianNetwork), "Stored model is not a BayesianModel."
        assert self.model.check_model(), "The Bayesian model is not valid."
        
        inference_engine = VariableElimination(self.model)
        results = {}
        for var in query_vars:
            if var not in self.model.nodes():
                # print(f"Variable {var} is not in the model.")
                results[var] = None
                continue
            table = inference_engine.query(variables=[var], evidence=evidence)
            # print(f"Variable {var}: ")
            # print(f"{table}")
            # print(f"Variable {var} should take value {table.values.argmax()}")
            results[var] = table.values.argmax()
        return results



def parse_hypothesis(hypothesis):
    if hypothesis == 'ABCconj':
        return ABCconj
    elif hypothesis == 'ABconj':
        return ABconj
    elif hypothesis == 'ACconj':
        return ACconj
    elif hypothesis == 'BCconj':
        return BCconj
    elif hypothesis == 'Adisj':
        return Adisj
    elif hypothesis == 'Bdisj':
        return Bdisj
    elif hypothesis == 'Cdisj':
        return Cdisj
    elif hypothesis == 'ABCdisj':
        return ABCdisj
    elif hypothesis == 'ABdisj':
        return ABdisj
    elif hypothesis == 'ACdisj':
        return ACdisj
    elif hypothesis == 'BCdisj':
        return BCdisj
    else:
        raise ValueError("Invalid hypothesis")

if __name__ == '__main__':
    # Create the environment
    hypothesis = parse_hypothesis(hypothesis)
    env = CausalEnv_v1({"reward_structure": "baseline", 
                        "quiz_disabled_steps": -1,
                        "hypotheses": [hypothesis],  # single hypothesis
                        "max_baseline_steps": 100})

    # Define the RL agent   
    agent = Agent(env, random_action=random_action)
    agent.train(episodes=5000)
    # Let's say you want to infer the state of variables 0, 1, 2 when variable 3 is True
    evidence = {3: 1}  # Assuming binary states where 1 represents True
    query_vars = [0, 1, 2]  # Variables you want to inquire about
    inferred_probs = agent.inference(query_vars, evidence)
    print(inferred_probs)

                
# Close the environment
env.close()