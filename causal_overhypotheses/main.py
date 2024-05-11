import numpy as np
import torch
import argparse

# parse args: seed, hypothesis, random_action
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hypothesis', type=str, default='ABCconj')
parser.add_argument('--random_action', type=bool, default=False)  # pass True to use random actions, don't pass anything to use RL actions
parser.add_argument('--exploration_scale', type=float, default=0.01)
parser.add_argument('--template_matching_scale', type=float, default=0.02)
args = parser.parse_args()
seed = args.seed
hypothesis = args.hypothesis
random_action = args.random_action
exploration_scale = args.exploration_scale
template_matching_scale = args.template_matching_scale
print(seed, hypothesis, random_action)
print(f"Exploration scale: {exploration_scale}, Template matching scale: {template_matching_scale}")
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
    def __init__(self, input_size, action_size, buffer_capacity=1000,
                convergence_threshold=5, hidden_size=128, device=device, random_action=False):
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.state_visit_counts = defaultdict(lambda: defaultdict(int))
        self.previous_graphs = deque(maxlen=convergence_threshold)
        self.convergence_threshold = convergence_threshold
        self.core = Core(input_size, action_size, hidden_size, device)
        self.hidden_size = hidden_size
        self.device = device
        self.random_action = random_action
        self.model = None
        self.template_actions = None
        self.total_episodes = 0
        self.phase = 'explore'  # start with explore

    def select_action(self, env, state, random_action=False):
        if random_action:
            action = env.action_space.sample()
            return action
        else:
            action, value, log_prob = self.core.select_action(state, self.phase)  # uses different policy in different phases
            self.state_visit_counts[tuple(state)][action] += 1
            return action, value, log_prob

    def compute_explore_intrinsic_reward(self, env, state, action, epsilon=0.001):
        visit_count = self.state_visit_counts[tuple(state)][action]
        intrinsic_reward = (env._steps / (visit_count + env._steps * epsilon))**0.5  # Zhang et al. (2018)
        return intrinsic_reward
    
    def calculate_match_score_to_template(self, action, env=None, state=None):
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

    def train(self, env, episodes, scale=0.001, inference_interval=10, evaluation_interval=10, evaluation_episodes=5, int_reward_function=None):
        assert int_reward_function is not None, "No reward function provided."
        losses = []
        for episode in range(episodes):
            state = env.reset()
            self.core.reset_hidden()
            done = False
            rewards, log_probs, values, dones = [], [], [], []
            while not done:
                if self.random_action:
                    action = self.select_action(env, state, self.random_action)
                    next_state, reward, done, _ = env.step(action)
                else:
                    action, value, log_prob = self.select_action(env, state, self.random_action)
                    env_action = integer_action_to_tuple(action, env._n_blickets)
                    next_state, reward, done, _ = env.step(env_action)
                    # convert action to one-hot vector if int_reward_function is calculate_match_score_to_template
                    if int_reward_function == self.calculate_match_score_to_template:
                        action = tuple_action_to_vector(env_action, env._n_blickets)
                    intrinsic_reward = int_reward_function(env=env, state=state, action=action)
                    total_reward = reward + scale * intrinsic_reward
                    rewards.append(total_reward)
                    log_probs.append(log_prob)
                    values.append(value)
                    dones.append(done)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state

            if not self.random_action:
                loss = self.core.learn(log_probs, values, rewards, dones, self.phase)
                losses.append(loss)

            # Evaluate the performance periodically
            if episode % evaluation_interval == 0 and episode > 0:
                self.total_episodes += evaluation_interval
                print(f"Phase: {self.phase}, episode {episode}. Total episode {self.total_episodes}.")
                self.evaluate_and_report(env, evaluation_episodes)
                if not self.random_action:
                    print(f"Episode {episode}. Running average loss: {sum(losses[-inference_interval:]) / 10:.2f}")
            
            if self.phase=="explore": 
                # Periodically infer the causal graph
                if episode % inference_interval == 0:
                    self.model = self.infer_causal_graph()
                    if self.model:
                        edges = frozenset(self.model.edges())
                        self.previous_graphs.append(edges)
                        print("Learned model structure:", edges)
                        if self.has_converged():
                            print(f"!!!Model has converged after {episode} episodes.")
                            self.phase = 'exploit'
                            break

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
    
    def evaluate_and_report(self, env, episodes):
        total_rewards = 0  # reward from environment
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                if self.random_action:
                    action = env.action_space.sample()
                    next_state, reward, done, _ = env.step(action)
                    total_rewards += reward
                    state = next_state
                else:
                    action, _, _ = self.core.select_action(state, phase=self.phase)  # exploit but don't count visitations
                    env_action = integer_action_to_tuple(action, env._n_blickets)
                    next_state, reward, done, _ = env.step(env_action)
                    total_rewards += reward
                    state = next_state
                    
        average_reward = total_rewards / episodes
        print(f"Evaluation over {episodes} episodes: Avg Reward = {average_reward:.2f}")
    
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

    def run_full_pipeline(self, env1, env2, query_vars, evidence, exploration_scale, template_matching_scale, exploration_episodes=1000, exploitation_episodes=1000):
        # Explore phase
        self.train(env1, exploration_episodes, scale=exploration_scale, evaluation_interval=10, evaluation_episodes=5,
                   int_reward_function=self.compute_explore_intrinsic_reward)

        # Infer template actions
        self.inference(query_vars=query_vars, evidence=evidence)

        # # Exploit phase
        self.train(env2, exploitation_episodes, scale=template_matching_scale, evaluation_interval=10, evaluation_episodes=5,
                   int_reward_function=self.calculate_match_score_to_template)
        




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
    agent = Agent(buffer_capacity=1000,
                  random_action=random_action, 
                  input_size=explore_env.observation_space.shape[0], 
                  action_size=explore_env._n_blickets*2, 
                  hidden_size=128, 
                  device=device)
    agent.run_full_pipeline(env1=explore_env, env2=quiz_env, query_vars=query_vars, evidence=evidence, 
        exploration_scale=exploration_scale,
        template_matching_scale=template_matching_scale,
        exploration_episodes=1000, exploitation_episodes=1000)
