import re
import pandas as pd
import numpy as np

output_file_path = "/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/bayes_4712295.out"

result_dict = {}  # keys: hypotheses. Values: dict with keys "random action" or "actor critic", and values list of number of episodes to convergence

hypothesis_list = ["ABCconj", "ABconj", "ACconj", "BCconj", "Adisj", "Bdisj", "Cdisj", "ABCdisj", "ABdisj", "ACdisj", "BCdisj"]

# Initialize result_dict with the hypotheses and methods
for hypothesis in hypothesis_list:
    result_dict[hypothesis] = {"random action": [], "actor critic": []}

# Variable to keep track of the current hypothesis and method
current_hypothesis = None
current_method = None
has_converged = False  # Tracker for convergence

with open(output_file_path, "r") as f:
    for line in f:
        # Check if the line contains a hypothesis and method
        match = re.match(r"(\d+)\s+(\w+)\s+(True|False)", line)
        if match:
            # If starting a new hypothesis/method, check if the last one converged
            if current_hypothesis and current_method and not has_converged:
                result_dict[current_hypothesis][current_method].append(5000)  # Append default value for non-convergence
            
            # Extract hypothesis and whether it's random action or actor critic
            _, hypothesis, is_random_action = match.groups()
            current_hypothesis = hypothesis
            current_method = "random action" if is_random_action == "True" else "actor critic"
            has_converged = False  # Reset the convergence tracker

        # Check if the line indicates convergence
        convergence_match = re.search(r"Model has converged after (\d+) episodes.", line)
        if convergence_match and current_hypothesis and current_method:
            # Extract number of episodes and append to the appropriate list in the dictionary
            episodes = int(convergence_match.group(1))
            result_dict[current_hypothesis][current_method].append(episodes)
            has_converged = True  # Set the convergence tracker

    # Check for the last hypothesis/method in the file
    if current_hypothesis and current_method and not has_converged:
        result_dict[current_hypothesis][current_method].append(5000)  # Append default value for non-convergence



# You can add print statements to verify the results
for hypothesis, methods in result_dict.items():
    print(f"{hypothesis}:")
    for method, episodes in methods.items():
        print(f"  {method}: {episodes}")

# Initialize an empty list to hold all rows of data
data_rows = []

# Loop over the dictionary to process each hypothesis and method
for hypothesis, methods in result_dict.items():
    for method, episodes in methods.items():
        # Calculate mean and standard deviation if there are episodes recorded
        if episodes:
            mean = np.mean(episodes)
            std_dev = np.std(episodes)
        else:
            mean = np.nan  # Use NaN for cases with no episodes
            std_dev = np.nan
        
        # Append each row of data to the list
        data_rows.append({'Hypothesis': hypothesis, 'Method': method, 'Mean': mean, 'Std Dev': std_dev})

# Create DataFrame from the list of dictionaries
df = pd.DataFrame(data_rows)

# Pivot the DataFrame for means and standard deviations separately
mean_table = df.pivot(index='Method', columns='Hypothesis', values='Mean')
std_table = df.pivot(index='Method', columns='Hypothesis', values='Std Dev')

# Print the tables, :2f limits the number of decimal places to 2
print("Mean Table:")
print(mean_table.to_string(float_format="{:.2f}".format))
print("\nStandard Deviation Table:")
print(std_table.to_string(float_format="{:.2f}".format))
