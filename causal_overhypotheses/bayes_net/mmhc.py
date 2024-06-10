import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore, MmhcEstimator
from pgmpy.models import BayesianNetwork
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import sys


# Load data
data = pd.read_csv('/home/mila/l/lindongy/causal_rl/causal_overhypotheses/ktd_6apples_1000epi.csv')
print(data.columns)
# data = np.array([[0,0,0],
#                  [1,0,0],
#                  [1,1,1],
#                  [1,1,0]])
print(data.head())
data = data.astype(int)
data = data.drop_duplicates()
print(len(data))  # number of unique observations
print(data)
n_repeats = 100  # could be as few as 4
data = data.reindex(data.index.repeat(n_repeats))
# shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# MMHC Algorithm
# print(data.head(10))
est = MmhcEstimator(data)
structure = est.estimate()
model = BayesianNetwork(structure.edges())
print(structure.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)
print(frozenset(model.edges()))

# Inference
# results = {}
# query_vars = [0, 1, 2, 3]
# evidence = {4:1}
# inference_engine = VariableElimination(model)
# for var in query_vars:
#     if var not in model.nodes():
#         # print(f"Variable {var} is not in the model.")
#         results[var] = None
#         continue
#     table = inference_engine.query(variables=[var], evidence=evidence)
#     print(f"Variable {var}: ")
#     print(f"{table}")
#     print(f"Variable {var} should take value {table.values.argmax()}")
#     results[var] = table.values.argmax()

# Visualize the graph
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
plt.savefig('mmhc_algo3.png')