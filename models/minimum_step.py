import torch
import numpy as np
import random


class Node:
    def __init__(self, belief, parent_action=None):
        self.belief = belief
        self.children = {}
        self.parent_action = parent_action

def expand_node(node, depth=1, max_depth=5):
    if depth > max_depth:
        return np.inf

    # If only one hypothesis is left or its probability is very high
    max_belief = torch.max(node.belief)
    if len(node.belief[node.belief > 1e-6]) == 1 or max_belief > 0.95:
        return depth

    # For each action, create a child node and compute its belief
    model = BayesianModel()
    model.prior = node.belief

    depths = []
    for action in BlicketEnv("").actions:
        future_belief = model.prior.clone()
        likelihoods = torch.tensor([BlicketEnv(h).step(action) for h in model.hypotheses], dtype=torch.float32)
        numerator = likelihoods * future_belief
        denominator = torch.sum(numerator)
        future_belief = numerator / denominator

        child_node = Node(future_belief, action)
        node.children[action] = child_node
        depths.append(expand_node(child_node, depth + 1, max_depth))

    # Return the minimum expected depth from this node
    return min(depths)

def best_policy_sequence():
    root = Node(torch.ones(len(BayesianModel().hypotheses)) / len(BayesianModel().hypotheses))
    expand_node(root)
    sequence = []

    current = root
    while current.children:
        action = min(current.children, key=lambda a: len(current.children[a].children))
        sequence.append(action)
        current = current.children[action]

    return sequence
