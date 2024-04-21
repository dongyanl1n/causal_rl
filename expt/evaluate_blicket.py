import torch
import numpy as np
import random
from models.maximum_info_gain import BayesianModel
from envs.blicket import BlicketEnv

def evaluate_per_step_model(env, model, num_episodes=1000, epsilon=0.1):
    """
    Evaluate the BayesianModel's ability to determine the true hypothesis.
    """

    # Track how many steps it takes for the model to become confident
    steps_to_converge = []

    for episode in range(num_episodes):
        model.prior = torch.ones(len(model.hypotheses)) / len(model.hypotheses)  # Reset belief
        converged = False
        step = 0
        while not converged and step < 100:  # Max of 20 steps for convergence
            action = model.select_action(epsilon)
            observation = env.step(action)
            model.update_belief(action, observation)

            step += 1

            # Check if belief for the true hypothesis exceeds a threshold (e.g., 0.9)
            true_hypothesis_index = model.hypotheses.index(env.true_hypothesis)
            if model.prior[true_hypothesis_index] > 0.9:
                converged = True

        steps_to_converge.append(step if converged else 100)

    return sum(steps_to_converge) / num_episodes  # Average steps to converge


def evaluate_tree_policy(env, model, max_steps=20):
    """
    Evaluate the tree-based policy's ability to determine the true hypothesis.
    """
    env = BlicketEnv(true_hypothesis)
    model = BayesianModel()

    action_sequence = best_policy_sequence()

    for step in range(min(max_steps, len(action_sequence))):
        action = action_sequence[step]
        observation = env.step(action)
        model.update_belief(action, observation)

        # Check if the belief for the true hypothesis exceeds a threshold (e.g., 0.9)
        true_hypothesis_index = model.hypotheses.index(true_hypothesis)
        if model.prior[true_hypothesis_index] > 0.9:
            return step + 1  # Return number of steps to determine the hypothesis

    return max_steps


def main():
    # Evaluate the per-step model
    per_step_model_results = []
    for true_hypothesis in ["A-dis", "B-dis", "C-dis", "AB-dis", "AC-dis", "BC-dis", "ABC-dis",
                            "AB-con", "AC-con", "BC-con", "ABC-con"]:
        env = BlicketEnv(true_hypothesis)
        model = BayesianModel()
        per_step_model_results.append(evaluate_per_step_model(env, model))
    print("Average steps to converge (per-step model):", sum(per_step_model_results) / len(per_step_model_results))

    # Evaluate the tree-based policy
    tree_policy_results = []
    for true_hypothesis in ["A-dis", "B-dis", "C-dis", "AB-dis", "AC-dis", "BC-dis", "ABC-dis",
                            "AB-con", "AC-con", "BC-con", "ABC-con"]:
        env = BlicketEnv(true_hypothesis)
        model = BayesianModel()  # TODO: Replace with tree-based policy
        tree_policy_results.append(evaluate_tree_policy(env, model))
    print("Average steps to converge (tree-based policy):", sum(tree_policy_results) / len(tree_policy_results))

if __name__ == "__main__":
    main()
