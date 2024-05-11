#!/bin/bash
#SBATCH --job-name=bayes
#SBATCH --output=/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/bayes_%j.out
#SBATCH --error=/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/bayes_%j.err
#SBATCH --partition=main
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec

module load python/3.7
source ~/envs/causal_rl/bin/activate

# python driver.py --alg ppo2 --policy mlp_lstm --lstm_units 512 --reward_structure quiz --holdout_strategy none

# python scripts/collect_trajectories.py --reward_structure quiz --quiz_disabled_steps 30 --max_steps 40 --num_trajectories 10000 

# Hypotheses array
hypotheses=("ABconj" "ACconj" "BCconj" "Adisj" "Bdisj" "Cdisj" "ABCdisj" "ABCconj" "ABdisj" "ACdisj" "BCdisj")

# Loop through all hypotheses
for hypo in "${hypotheses[@]}"; do
    # Loop through seeds 0-9
    for seed in {0..9}; do
        # Run without random action
        python main.py --seed "$seed" --hypothesis "$hypo" --template_matching_scale 0.05
        # Run with random action
        python main.py --seed "$seed" --hypothesis "$hypo" --random_action True --template_matching_scale 0.05
    done
done


