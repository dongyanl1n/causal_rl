import gym
from envs.causal_env_v0 import CausalEnv_v1, ABconj, ACconj, BCconj, Adisj, Bdisj, Cdisj, ABCdisj
from models.replay_buffer import ReplayBuffer
import pandas as pd
from pgmpy.estimators import MmhcEstimator
from collections import defaultdict
from collections import deque
from tqdm import tqdm


# Create the environment
env = CausalEnv_v1({"reward_structure": "baseline", 
                    "quiz_disabled_steps": -1,
                    "hypotheses": [ABCdisj],  # single hypothesis
                    "max_baseline_steps": 100})


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


# Define the RL agent
class Agent:
    def __init__(self, env, buffer_capacity=1000, batch_size=10, convergence_threshold=5):
        self.env = env
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.state_visit_counts = defaultdict(lambda: defaultdict(int))
        self.previous_graphs = deque(maxlen=convergence_threshold)
        self.convergence_threshold = convergence_threshold

    def infer_causal_graph(self):
        data = [entry[0] for entry in self.replay_buffer.get_all_data()]  # Assume states are stored in the first index
        if len(data) < 1:
            return None
        mmhc = MmhcEstimator(pd.DataFrame(data))
        model = mmhc.estimate()
        return model

    def select_action(self, state):
        action = self.env.action_space.sample()  # Random action; replace with your policy
        self.state_visit_counts[tuple(state)][action] += 1
        return action

    def compute_intrinsic_reward(self, state, action):
        visit_count = self.state_visit_counts[tuple(state)][action]
        intrinsic_reward = 1 / (visit_count + 1)  # Simple count-based intrinsic reward
        return intrinsic_reward

    def has_converged(self):
        if len(self.previous_graphs) < self.convergence_threshold:
            return False
        return all(g == self.previous_graphs[0] for g in self.previous_graphs)

    def train(self, episodes):
        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                intrinsic_reward = self.compute_intrinsic_reward(state, action)
                total_reward = reward + intrinsic_reward
                self.replay_buffer.push(state, action, total_reward, next_state, done)
                state = next_state

            # Periodically infer the causal graph
            if episode % 10 == 0:
                model = self.infer_causal_graph()
                if model:
                    edges = frozenset(model.edges())
                    self.previous_graphs.append(edges)
                    print("Learned model structure:", edges)
                    if self.has_converged():
                        print(f"Model has converged after {episode} episodes.")
                        break
            # Perform any necessary logging or evaluation here

agent = Agent(env)
agent.train(episodes=500)

                
# Close the environment
env.close()