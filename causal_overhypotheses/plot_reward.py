import numpy as np
import matplotlib.pyplot as plt
import sys
import re
import pandas as pd
# Define the path to the SLURM output file
output_file_path = "/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/bayes-multienv-causalnet-10000_4730744.out"

# Function to extract the dictionary from the SLURM output file
def extract_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    # Regular expression to find the dictionary
    dict_pattern = re.compile(r'\{[0-9, \.\-:]+\}')
    dict_match = dict_pattern.search(content)
    
    if dict_match:
        dict_str = dict_match.group()
        result_dict = eval(dict_str)
        return result_dict
    else:
        raise ValueError("No dictionary found in the file.")

# Function to compute exponential moving average
def exponential_moving_average(data, span):
    ema = pd.Series(data).ewm(span=span, adjust=False).mean().to_numpy()
    return ema

# Extract the dictionary
results_dict = extract_dict_from_file(output_file_path)

# Extract keys and values
episodes = list(results_dict.keys())
average_rewards = list(results_dict.values())

# Compute smoothed average rewards using EMA
span = 50  # You can adjust the span for smoothing
smoothed_rewards_ema = exponential_moving_average(average_rewards, span)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(episodes, average_rewards, marker='o', linestyle='-', label='Average Rewards')
plt.plot(episodes, smoothed_rewards_ema, marker='', linestyle='-', label='Smoothed Average Rewards (EMA)', color='red')
plt.xlabel('Episodes')
plt.ylabel('Average Rewards')
plt.title('Average Rewards vs Episodes')
plt.grid(True)
plt.savefig('reward_plot_causalnet_tm0.png')
