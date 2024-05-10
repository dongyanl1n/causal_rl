import numpy as np
import torch
import argparse

# parse args: seed, hypothesis, random_action
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hypothesis', type=str, default='ABCconj')
parser.add_argument('--random_action', type=bool, default=False)  # pass True to use random actions, don't pass anything to use RL actions
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



def integer_action_to_tuple(action, n_blickets):
    # here, action is a flattened vector of dimension n_blickets*2
    # 0 -> (0,0)
    # 1 -> (1,0)
    # 2 -> (2,0)
    # 3 -> (0,1)
    # 4 -> (1,1)
    # 5 -> (2,1)
    return (action % n_blickets, action // n_blickets)

def tuple_action_to_vector(action, n_blickets):
    # (0,0) -> 0 -> [1,0,0,0,0,0]
    # (1,0) -> 1 -> [0,1,0,0,0,0]
    # (2,0) -> 2 -> [0,0,1,0,0,0]
    # (0,1) -> 3 -> [0,0,0,1,0,0]
    # (1,1) -> 4 -> [0,0,0,0,1,0]
    # (2,1) -> 5 -> [0,0,0,0,0,1]
    return np.eye(n_blickets*2)[action[0] + action[1]*n_blickets]




# Define the RL agent
class Agent:
    def __init__(self, input_size, action_size, buffer_capacity=1000, batch_size=10, 
                convergence_threshold=5, hidden_size=128, device=device, random_action=False):
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.state_visit_counts = defaultdict(lambda: defaultdict(int))
        self.previous_graphs = deque(maxlen=convergence_threshold)
        self.convergence_threshold = convergence_threshold
        self.explorer = Core(input_size, action_size, hidden_size, device)
        self.template_matcher = Core(input_size, action_size, hidden_size, device)
        self.hidden_size = hidden_size
        self.device = device
        self.random_action = random_action
        self.model = None
        self.template_actions = None
        self.total_episodes = 0

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

    def select_action(self, env, state, random_action=False):
        if random_action:
            action = env.action_space.sample()
            return action
        else:
            action, value, log_prob = self.explorer.select_action(state)
            self.state_visit_counts[tuple(state)][action] += 1
            return action, value, log_prob

    def compute_explore_intrinsic_reward(self, env, state, action, epsilon=0.001):
        visit_count = self.state_visit_counts[tuple(state)][action]
        intrinsic_reward = (env._steps / (visit_count + env._steps * epsilon))**0.5  # Zhang et al. (2018)
        return intrinsic_reward
    
    def calculate_match_score_to_template(self, action):
        r_total = 0
        for template_action in self.template_actions:
            # calculate cosine similarity between action (one-hot vector) and template_action (one-hot vector)
            assert len(action) == len(template_action), "Action and template action have different lengths."
            r = np.dot(action, template_action) / (np.linalg.norm(action) * np.linalg.norm(template_action))
            r_total += r
        return r_total

    def has_converged(self):
        if len(self.previous_graphs) < self.convergence_threshold:
            return False
        return all(g == self.previous_graphs[0] for g in self.previous_graphs)

    def explore(self, env, episodes, scale=0.001, inference_interval=10):
        losses = []
        for episode in range(episodes):
            state = env.reset()
            self.explorer.reset_hidden()
            done = False
            rewards, log_probs, values, dones = [], [], [], []
            while not done:
                if self.random_action:
                    action = self.select_action(env, state, random_action=self.random_action)
                    next_state, reward, done, _ = env.step(action)
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                else:
                    action, value, log_prob = self.select_action(env, state)  # action comes out an integer
                    env_action = integer_action_to_tuple(action, env._n_blickets)
                    next_state, reward, done, _ = env.step(env_action)
                    intrinsic_reward = self.compute_explore_intrinsic_reward(env, state, action)
                    total_reward = reward + scale * intrinsic_reward
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    rewards.append(total_reward)
                    log_probs.append(log_prob)
                    values.append(value)
                    dones.append(done)
                    state = next_state

            if not self.random_action:
                loss = self.explorer.learn(log_probs, values, rewards, dones)
                losses.append(loss)

            # Periodically infer the causal graph
            if episode % inference_interval == 0:
                self.model = self.infer_causal_graph()
                if self.model:
                    edges = frozenset(self.model.edges())
                    self.previous_graphs.append(edges)
                    print("Learned model structure:", edges)
                    self.total_episodes += inference_interval
                    if self.has_converged():
                        print(f"Model has converged after {episode} episodes.")
                        break
                if not self.random_action:
                    print(f"Episode {episode}. Running average loss: {sum(losses[-inference_interval:]) / 10:.2f}")


    def inference(self, query_vars, evidence):
        """
        Perform inference given evidence and return the probabilities for query variables, and translate them to template actions.
        
        :param query_vars: List of variable indices for which to compute probabilities
        :param evidence: Dictionary of observed variables and their values
        :return: A dictionary of probability distributions for each query variable, and a list of template actions
        """
        assert self.model is not None, "Model has not been learned yet."
        assert self.has_converged(), "Model has not converged yet."
        assert isinstance(self.model, BayesianNetwork), "Stored model is not a BayesianModel."
        assert self.model.check_model(), "The Bayesian model is not valid."
        
        inference_engine = VariableElimination(self.model)
        results = {}
        self.template_actions = []
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
            self.template_actions.append(tuple_action_to_vector(tuple([var, table.values.argmax()]), n_blickets=3))
        return results, self.template_actions
    

    def template_matching(self, env, episodes, scale=0.02, evaluation_interval=10, evaluation_episodes=5):
        # train self.template_matcher to maximize the match score to the template actions
        # same as explore, but with intrinsic_reward = calculate_match_score_to_template(action, template_actions)
        losses = []
        for episode in range(episodes):
            state = env.reset()
            self.template_matcher.reset_hidden()
            done = False
            rewards, log_probs, values, dones = [], [], [], []
            while not done:
                if self.random_action:
                    action = env.action_space.sample() # no longer needs to count
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                else:
                    action, value, log_prob = self.template_matcher.select_action(state)
                    env_action = integer_action_to_tuple(action, env._n_blickets)
                    next_state, reward, done, _ = env.step(env_action)
                    intrinsic_reward = self.calculate_match_score_to_template(torch.eye(env._n_blickets*2)[action])
                    total_reward = reward + scale * intrinsic_reward
                    rewards.append(total_reward)
                    log_probs.append(log_prob)
                    values.append(value)
                    dones.append(done)
                    state = next_state

            if not self.random_action:
                loss = self.template_matcher.learn(log_probs, values, rewards, dones)
                losses.append(loss)

            # Evaluate every `evaluation_interval` episodes
            if episode % evaluation_interval == 0:
                self.evaluate_and_report(env, evaluation_episodes)
                self.total_episodes += evaluation_interval
                if not self.random_action:
                    print(f"Episode {episode}. Running average loss: {sum(losses[-evaluation_interval:]) / evaluation_interval:.2f}")

        print(f"Training complete. Finished training after {self.total_episodes} episodes.")

    def evaluate_and_report(self, env, episodes):
        total_rewards = 0  # reward from environment
        match_scores = 0
        for _ in range(episodes):
            state = env.reset()
            done = False
            t = 0
            while not done:
                t += 1
                if self.random_action:
                    action = env.action_space.sample()
                    next_state, reward, done, _ = env.step(action)
                    intrinsic_reward = self.calculate_match_score_to_template(tuple_action_to_vector(action, env._n_blickets))
                    total_rewards += reward
                    match_scores += intrinsic_reward
                    state = next_state
                else:
                    action, _, _ = self.template_matcher.select_action(state)
                    env_action = integer_action_to_tuple(action, env._n_blickets)
                    next_state, reward, done, _ = env.step(env_action)
                    intrinsic_reward = self.calculate_match_score_to_template(torch.eye(env._n_blickets*2)[action])
                    total_rewards += reward
                    match_scores += intrinsic_reward
                    state = next_state
                    

        average_reward = total_rewards / episodes
        average_match_score = match_scores / episodes
        print(f"Evaluation over {episodes} episodes: Avg Reward = {average_reward:.2f}, Avg Match Score = {average_match_score:.2f}")
        




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
    explore_env = CausalEnv_v1({"reward_structure": "baseline", 
                        "hypotheses": [hypothesis],  # single hypothesis
                        "max_steps": 100})

    quiz_env = CausalEnv_v1({"reward_structure": "quiz", 
                        "hypotheses": [hypothesis],  # single hypothesis
                        "max_steps": 5,
                        "add_step_reward_penalty": True})
    evidence = {3: 1}  # Want detector to be True
    query_vars = [0, 1, 2]  # Want to infer values of blickets A,B,C

    # Define the RL agent   
    agent = Agent(random_action=random_action, 
                  input_size=explore_env.observation_space.shape[0], 
                  action_size=explore_env._n_blickets*2, 
                  hidden_size=128, 
                  device=device)
    agent.explore(explore_env, episodes=5000, scale=0.001, inference_interval=10)
    agent.inference(query_vars, evidence)
    agent.template_matching(quiz_env, episodes=1000, scale=0.02, evaluation_interval=100, evaluation_episodes=10)
