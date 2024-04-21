# Bayesian model to maximize expected per-action information gain
import torch
import numpy as np
import random


def likelihood_function(hypothesis, action, observation):
    if hypothesis == "A-dis":
        if action in ["A", "AB", "AC", "ABC"] and observation == 1:
            return 1
        elif action in ["B", "C", "BC"] and observation == 0:
            return 1
        else:
            return 0

    # TODO: Fill in the rest of the likelihood function

    elif hypothesis == "ABC-con":
        if action == "ABC" and observation == 1:
            return 1
        elif action in ["", "A", "B", "C", "AB", "AC", "BC"] and observation == 0:
            return 1
        else:
            return 0

    return 0


def information_gain(prior, posterior):
    """
    Compute the KL divergence between prior and posterior.
    KL(prior || posterior)
    """
    # Only consider non-zero entries to avoid NaNs from log(0)
    mask = prior * posterior > 0
    kl_div = torch.sum(prior[mask] * torch.log(prior[mask] / posterior[mask]))
    return kl_div


class MaxInfoGainModel:
    def __init__(self):
        # Define all possible hypotheses
        self.hypotheses = [
            "A-dis", "B-dis", "C-dis", "AB-dis", "AC-dis", "BC-dis", "ABC-dis",
            "AB-con", "AC-con", "BC-con", "ABC-con"
        ]
        # Define all possible actions
        self.actions = ["A", "B", "C", "AB", "BC", "AC", "ABC"]
        # Initialize with uniform prior
        self.prior = torch.ones(len(self.hypotheses)) / len(self.hypotheses)

    def select_action(self, observation, epsilon=0.1):
        expected_information_gains = []
        for action in self.actions:
            gains = []
            for h in self.hypotheses:
                new_belief = self.compute_posterior(action, observation)
                gain = information_gain(self.prior, new_belief)
                gains.append(gain)
            expected_information_gains.append(torch.mean(torch.tensor(gains)))

        if torch.rand(1) < epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[torch.argmax(torch.tensor(expected_information_gains))]

    def compute_posterior(self, action, observation):
        likelihoods = torch.tensor([likelihood_function(h, action, observation) for h in self.hypotheses],
                                   dtype=torch.float32)
        numerator = likelihoods * self.prior
        denominator = torch.sum(numerator)
        return numerator / denominator

    def update_belief(self, action, observation):
        self.prior = self.compute_posterior(action, observation)
