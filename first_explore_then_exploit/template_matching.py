import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_file_path = "/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/bayes_4716503.out"

# Initialize the result dictionary
hypothesis_list = ["ABCconj", "ABconj", "ACconj", "BCconj", "Adisj", "Bdisj", "Cdisj", "ABCdisj", "ABdisj", "ACdisj", "BCdisj"]
result_dict = {hyp: {"random action": {}, "actor critic": {}} for hyp in hypothesis_list}

def parse_log_file(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
    
    entries = data.split('Training complete.')
    for entry in entries:
        lines = entry.split('\n')
        if len(lines) < 3:
            continue
        
        seed_info = re.search(r'(\d+) (\w+) (True|False)', lines[1])
        if not seed_info:
            continue
        
        seed, condition, random_action = int(seed_info.group(1)), seed_info.group(2), seed_info.group(3) == 'True'
        method = "random action" if random_action else "actor critic"
        
        # Initialize episode number for random action
        episode_number = 0
        
        # Process each line for evaluation metrics
        result_dict[condition][method][seed] = {
            "episode_number": [],
            "avg_reward": [],
            "avg_match_score": []
        }

        for i in range(len(lines)):
            if lines[i].startswith('Evaluation over'):
                avg_reward = float(re.search(r'Avg Reward = ([\-\d\.]+)', lines[i]).group(1))
                avg_match_score = float(re.search(r'Avg Match Score = ([\-\d\.]+)', lines[i]).group(1))
                
                if not random_action:
                    # For actor critic, get the episode number from the next line
                    episode_number = int(re.search(r'Episode (\d+)', lines[i + 1]).group(1))
                result_dict[condition][method][seed]["episode_number"].append(episode_number)
                result_dict[condition][method][seed]["avg_reward"].append(avg_reward)
                result_dict[condition][method][seed]["avg_match_score"].append(avg_match_score)
                
                if random_action:
                    # Increase episode number by 50 for each evaluation for random action
                    episode_number += 50

# Call the parsing function
parse_log_file(output_file_path)

### Step 2: Plotting the Data
def plot_data(result_dict, condition):
    # Initialize a new figure
    plt.figure()
    colors = {'random action': 'blue', 'actor critic': 'red'}
    
    # Create the first axes (for rewards)
    ax1 = plt.gca()  # Get the current axes
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward')
    
    # Create a second axes that shares the same x-axis (for match scores)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Match Score')

    for method in ["random action", "actor critic"]:
        data = result_dict[condition][method]
        episode_numbers = data[0]["episode_number"]  # x axis
        avg_rewards = np.array([seed_data["avg_reward"] for seed_data in data.values()])
        avg_match_scores = np.array([seed_data["avg_match_score"] for seed_data in data.values()])

        # Calculate mean and standard deviation for rewards
        mean_rewards = np.mean(avg_rewards, axis=0)
        std_rewards = np.std(avg_rewards, axis=0)
        
        # Plot rewards
        ax1.plot(episode_numbers, mean_rewards, label=f'{method} Rewards', color=colors[method])
        ax1.fill_between(episode_numbers, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, color=colors[method])
        
        # Calculate mean and standard deviation for match scores
        mean_match_scores = np.mean(avg_match_scores, axis=0)
        std_match_scores = np.std(avg_match_scores, axis=0)
        
        # Plot match scores
        ax2.plot(episode_numbers, mean_match_scores, label=f'{method} Match Scores', linestyle='dashed', color=colors[method])
        ax2.fill_between(episode_numbers, mean_match_scores - std_match_scores, mean_match_scores + std_match_scores, alpha=0.2, color=colors[method])

    # Legends and titles
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'Performance in {condition}')

    # Save the plot to file
    plt.savefig(f'{condition}.png')
    plt.close()

### Step 3: Generate Plots for Each Condition
for condition in hypothesis_list:
    plot_data(result_dict, condition)

