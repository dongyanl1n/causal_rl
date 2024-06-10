import pandas as pd
from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load your data
# data = pd.read_csv('/Users/dongyanlin/Desktop/4blicket/ABdisj.csv')
# data.columns = ['A', 'B', 'C', 'D', 'Detector']
data = np.array([[0,0,0],
                 [1,0,0],
                 [1,1,1],
                 [1,1,0]])
data = pd.DataFrame(data, columns=['K', 'D', 'R'])
data = data.drop_duplicates()
# delete a row of data
# data = data.drop(data.index[0])
# repeat each observation different times
# repeats = np.random.randint(low=5, high=10, size=len(data))
# data = data.loc[np.repeat(data.index.values, repeats)]
n_repeats = 100  # could be as few as 4
data = data.reindex(data.index.repeat(n_repeats))
print(data)
# Apply the PC algorithm
pc = PC(data)
structure = pc.estimate()

# Convert the skeleton to a Bayesian Model
model = BayesianNetwork(structure.edges())
print(structure.edges())
# Fit the model using Bayesian Estimator
model.fit(data, estimator=BayesianEstimator)
# Check the learned model
print(frozenset(model.edges()))

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
plt.savefig('pc_algo.png')
