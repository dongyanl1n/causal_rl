#!/bin/bash
#SBATCH --job-name=bayes-multienv-5b-template_states-baseline
#SBATCH --output=/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/bayes-multienv-5b-template-states-baseline_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/bayes-multienv-5b-template-states-baseline_%A_%a.err
#SBATCH --partition=long
#SBATCH --mem=4G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec
#SBATCH --array=0-5

module load python/3.7
source ~/envs/causal_rl/bin/activate

# Define lists of hyperparameters
learning_rates=(0.01 0.001)
template_matching_scales=(0)
explore_scales=(0.01 0.1 0)
random_explores=(False)

# Calculate indices
lr_idx=$(( ($SLURM_ARRAY_TASK_ID / (${#template_matching_scales[@]} * ${#explore_scales[@]} * ${#random_explores[@]})) % ${#learning_rates[@]} ))
tm_idx=$(( ($SLURM_ARRAY_TASK_ID / (${#explore_scales[@]} * ${#random_explores[@]})) % ${#template_matching_scales[@]} ))
es_idx=$(( ($SLURM_ARRAY_TASK_ID / ${#random_explores[@]}) % ${#explore_scales[@]} ))
re_idx=$(( $SLURM_ARRAY_TASK_ID % ${#random_explores[@]} ))

# Select hyperparameters based on indices
lr=${learning_rates[$lr_idx]}
tm=${template_matching_scales[$tm_idx]}
es=${explore_scales[$es_idx]}
re=${random_explores[$re_idx]}

# Construct the command
command="python main.py --n_blickets 5 --model baseline --lr $lr --total_n_epi 150000 --episodes_per_env 500 --explore_scale $es"


echo "Running command: $command"
eval $command
