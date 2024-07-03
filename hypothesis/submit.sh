#!/bin/bash
#SBATCH --job-name=fourrooms-transfer
#SBATCH --output=/network/scratch/l/lindongy/grid_blickets/sbatch_log/fourrooms-transfer_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/grid_blickets/sbatch_log/fourrooms-transfer_%A_%a.err
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --array=1-16

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

# Define hyperparameters
learning_rates=(0.0001 0.0005)
seeds=(1 2 3 4)
scripts=("baseline_transfer.py" "main_transfer.py")

# Calculate hyperparameter indices
lr_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % 2 ))
seed_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / 2 % 4 ))
script_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / 8 ))

# Set hyperparameters
lr=${learning_rates[$lr_idx]}
seed=${seeds[$seed_idx]}
script=${scripts[$script_idx]}

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Running job with lr=$lr, seed=$seed, and script=$script"

# Run the command with the specified hyperparameters
python -u $script --env-name MiniGrid-FourRooms-v0 --lr $lr --seed $seed --recurrent-policy
