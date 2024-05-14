import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import random
from tqdm import tqdm

# parse args: seed, hypothesis, random_action
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--explore_scale', type=float, default=0.001)
parser.add_argument('--template_matching_scale', type=float, default=0.02)
parser.add_argument('--random_explore', type=bool, default=False)
args = parser.parse_args()
seed = args.seed
explore_scale = args.explore_scale
template_matching_scale = args.template_matching_scale
random_explore = args.random_explore
print(f"Seed: {seed}")
print(f"Random explore: {random_explore}")
print(f"Explore scale: {explore_scale}")
print(f"Template matching scale: {template_matching_scale}")
# Set seed for reproducibility
np.random.seed(seed)  # Set seed for numpy
torch.manual_seed(seed)  # Set seed for torch
torch.cuda.manual_seed(seed)  # Set seed for current CUDA device
random.seed(seed)  # Set seed for Python random module

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
class MultiEnvAgent:
    def __init__(self, input_size, action_size, buffer_capacity=1000,
                convergence_threshold=5, hidden_size=128, device=device):
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.state_visit_counts = defaultdict(lambda: defaultdict(int))
        self.previous_graphs = deque(maxlen=convergence_threshold)
        self.convergence_threshold = convergence_threshold
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.device = device
        self.core = Core(self.input_size, self.action_size, self.hidden_size, self.device)
        self.model = None # bayes net
        self.template_actions = None
        self.total_episode_counter = 0
        self.episodes_since_last_env = 0
        self.phase = 'explore'  # start with explore
        self.env = None
        self.hypothesis_list = ['Adisj', 'Bdisj', 'Cdisj', 'ABCdisj', 'ABdisj', 'ACdisj', 'BCdisj']
        # self.test_hypothesis_list = ['ABCconj', 'ABconj', 'ACconj', 'BCconj']

    def reinit_for_env(self):
        self.reinit_graphs()
        self.reinit_buffer()
        self.core.reset_hidden()
        self.episodes_since_last_env = 0
        self.phase = 'explore'

    def reinit_graphs(self):
        self.previous_graphs = deque(maxlen=self.convergence_threshold)
        self.model = None

    def reinit_buffer(self):
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer.capacity)
        self.state_visit_counts = defaultdict(lambda: defaultdict(int))
        
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

    
    def train(self, total_episodes, episodes_per_env, random_explore, 
              query_vars=[0, 1, 2], evidence={3:1}, 
              explore_scale=None, template_matching_scale=0.02,
              inference_interval=1, evaluation_interval=10, evaluation_episodes=5):
        results = {}
        gts = {}
        if not random_explore:
            assert explore_scale is not None, "Exploration scale must be provided."
        while self.total_episode_counter < total_episodes:
            if self.total_episode_counter % episodes_per_env == 0:  # Switch environment every episodes_per_env episodes
                print(f"Episode {self.total_episode_counter} of {total_episodes}. ")
                self.reinit_for_env()
                self.env = generate_env(self.hypothesis_list, max_steps=5)
                gts[self.total_episode_counter] = str(self.env._current_gt_hypothesis).split("'")[1].split('.')[-1]

            self.episodes_since_last_env += 1
            self.total_episode_counter += 1
            losses = []  # for logging

            if self.phase == 'explore':
                state = self.env.reset()
                # self.core.reset_hidden()
                done = False
                rewards, log_probs, values, dones = [], [], [], []

                while not done:
                    if random_explore:
                        action = self.select_action(self.env, state, random_explore)
                        next_state, reward, done, _ = self.env.step(action)
                    else:
                        action, value, log_prob = self.select_action(self.env, state, random_explore)
                        env_action = integer_action_to_tuple(action, self.env._n_blickets)
                        next_state, reward, done, _ = self.env.step(env_action)
                        intrinsic_reward = self.compute_explore_intrinsic_reward(env=self.env, state=state, action=action)
                        total_reward = reward + explore_scale * intrinsic_reward
                        rewards.append(total_reward)
                        log_probs.append(log_prob)
                        values.append(value)
                        dones.append(done)
                    
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state

                if not random_explore:
                    loss = self.core.learn(log_probs, values, rewards, dones, self.phase, retain_graph=True)  # calls loss.backward
                    losses.append(loss)

                if self.episodes_since_last_env % inference_interval == 0:
                    self.model = self.infer_causal_graph()
                    if self.model:
                        edges = frozenset(self.model.edges())
                        self.previous_graphs.append(edges)
                        # print("Learned model structure:", edges)
                        if self.has_converged():
                            print(f"Model has converged after {self.episodes_since_last_env} episodes.")
                            convergence_episode = self.episodes_since_last_env
                            _, _ = self.inference(query_vars=query_vars, evidence=evidence)
                            print("Final model structure:")
                            print(self.model.edges())
                            self.phase = 'exploit'

            elif self.phase == 'exploit':
                if self.episodes_since_last_env < episodes_per_env:
                    state = self.env.reset()
                    # self.core.reset_hidden()
                    done = False
                    rewards, log_probs, values, dones = [], [], [], []
                    while not done:
                        action, value, log_prob = self.select_action(self.env, state)
                        env_action = integer_action_to_tuple(action, self.env._n_blickets)
                        next_state, reward, done, _ = self.env.step(env_action)
                        intrinsic_reward = self.calculate_match_score_to_template(torch.eye(self.env._n_blickets*2)[action])
                        total_reward = reward + template_matching_scale * intrinsic_reward
                        rewards.append(total_reward)
                        log_probs.append(log_prob)
                        values.append(value)
                        dones.append(done)
                        state = next_state
                    
                    loss = self.core.learn(log_probs, values, rewards, dones, self.phase, retain_graph=True)  # calls loss.backward
                    losses.append(loss)
                    
                    if (self.episodes_since_last_env - convergence_episode) % evaluation_interval == 0:
                        average_reward = self.evaluate_and_report(self.env, evaluation_episodes, random_action=False)
                        results[self.total_episode_counter] = average_reward
        
            # if self.total_episode_counter % (3*episodes_per_env) == 0: # TODO: param 3
            #     print("========== Evaluating on unseen environments...==========")
            #     rewards = self.eval_on_unseen_env(episodes_per_env, random_explore, query_vars, evidence, explore_scale, template_matching_scale, inference_interval, evaluation_interval, evaluation_episodes)
        return results, gts


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
    
    def evaluate_and_report(self, env, episodes, random_action):
        total_rewards = 0  # reward from environment
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                if random_action:
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
        print(f"{self.episodes_since_last_env} episodes since last env. Avg Reward over {episodes} episodes = {average_reward:.2f}")
        return average_reward
    
    def infer_causal_graph(self):
        data = pd.DataFrame([entry[0] for entry in self.replay_buffer.get_all_data()])
        if data.empty:
            return None
        mmhc_estimator = MmhcEstimator(data)
        structure = mmhc_estimator.estimate()
        model = BayesianNetwork(structure.edges())
        # Here we need to fit the CPDs using, for example, Maximum Likelihood Estimation
        model.fit(data, estimator=MaximumLikelihoodEstimator)
        self.model = model
        return self.model
    

    # def eval_on_unseen_env(self, episodes_per_env, random_explore, query_vars, evidence, explore_scale, template_matching_scale, inference_interval, evaluation_interval, evaluation_episodes):
    #     env = generate_env(self.test_hypothesis_list, max_steps=5)
    #     self.reinit_graphs()
    #     self.reinit_buffer()
    #     self.phase = 'explore'  # Start in explore phase
    #     # self.core = Core(self.input_size, self.action_size, self.hidden_size, self.device)
    #     episode = 0
    #     all_rewards = []
    #     convergence_episode = None

    #     while episode < episodes_per_env:
    #         state = env.reset()
    #         self.core.reset_hidden()
    #         done = False
    #         episode_rewards = []

    #         while not done:
    #             if self.phase == 'explore':
    #                 if random_explore:
    #                     action = env.action_space.sample()
    #                     next_state, reward, done, _ = env.step(action)
    #                     episode_rewards.append(reward)
    #                 else:
    #                     action, value, log_prob = self.select_action(env, state, random_explore)
    #                     env_action = integer_action_to_tuple(action, env._n_blickets)
    #                     next_state, reward, done, _ = env.step(env_action)
    #                     intrinsic_reward = self.compute_explore_intrinsic_reward(env, state, action)
    #                     total_reward = reward + explore_scale * intrinsic_reward
    #                     episode_rewards.append(reward)

    #                 self.replay_buffer.push(state, action, reward, next_state, done)
    #                 state = next_state

    #                 if episode % inference_interval == 0:
    #                     self.model = self.infer_causal_graph()
    #                     if self.model:
    #                         edges = frozenset(self.model.edges())
    #                         self.previous_graphs.append(edges)
    #                         print("Learned model structure:", edges)
    #                         if self.has_converged():
    #                             self.phase = 'exploit'  # Switch to exploit phase once the model has converged
    #                             print(f"!!!Model has converged after {episode} episodes.")
    #                             convergence_episode = episode
    #                             _, _ = self.inference(query_vars=query_vars, evidence=evidence)
    #                             print("Final model structure:")
    #                             print(self.model.edges())
    #                             self.phase = 'exploit'

    #             elif self.phase == 'exploit':
    #                 if episode < episodes_per_env:
    #                     state = self.env.reset()
    #                     self.env._add_step_reward_penalty = True
    #                     self.env._add_quiz_positive_reward = True
    #                     self.core.reset_hidden()
    #                     done = False
    #                     while not done:
    #                         action, value, log_prob = self.select_action(self.env, state)
    #                         env_action = integer_action_to_tuple(action, self.env._n_blickets)
    #                         next_state, reward, done, _ = self.env.step(env_action)
    #                         intrinsic_reward = self.calculate_match_score_to_template(torch.eye(self.env._n_blickets*2)[action])
    #                         total_reward = reward + template_matching_scale * intrinsic_reward
    #                         episode_rewards.append(reward)
    #                         state = next_state

    #         all_rewards.append(sum(episode_rewards) / len(episode_rewards))  # Compute average reward for this episode

    #         episode += 1

    #     print("Evaluation on unseen environment completed.")
    #     return all_rewards


        
def generate_env(hypothesis_list, max_steps):

    hypothesis = random.choice(hypothesis_list)
    print(f"New environment with hypothesis: {hypothesis}")
    if type(hypothesis) == str:
        hypothesis = parse_hypothesis(hypothesis)

    # Create environments based on the phase: exploration or exploitation
    env = CausalEnv_v1({
        "hypotheses": [hypothesis],  # single hypothesis
        "max_steps": max_steps,
        'add_quiz_positive_reward': True})

    return env



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
    total_n_epi = 10000
    episodes_per_env = 500
    evidence = {3: 1}  # Want detector to be True
    query_vars = [0, 1, 2]  # Want to infer values of blickets A,B,C

    # Define the RL agent   
    agent = MultiEnvAgent(buffer_capacity=1000000,
                  input_size=4, # n_blickets + 1
                  action_size=6, # n_blickets * 2
                  hidden_size=128, 
                  device=device)

    # training
    results, gts = agent.train(total_episodes=total_n_epi, episodes_per_env=episodes_per_env,
                random_explore=random_explore, query_vars=query_vars, evidence=evidence, explore_scale=explore_scale,
                template_matching_scale=template_matching_scale, inference_interval=5, evaluation_interval=50, evaluation_episodes=10)
    print(results)
    print(gts)
    # plot results
    plt.plot(list(results.keys()), list(results.values()))
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    exp = 'random' if random_explore else 'exp{}'.format(explore_scale)
    plt.savefig(f'results_seed{seed}_tm{template_matching_scale}_{exp}.png')