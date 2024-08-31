#!/bin/bash
#SBATCH --job-name=6x6-2keys-best
#SBATCH --output=/network/scratch/l/lindongy/grid_blickets/sbatch_log/6x6-2keys-best_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/grid_blickets/sbatch_log/6x6-2keys-best_%A_%a.err
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --array=0-59

# This will run 60 combinations (6 hyperparameter combinations * 10 seeds each)

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

# Define parameter arrays
lr_values=(0.0006)
intrinsicR_scale_values=(0.2 0.3)
lrConSpec_values=(0.01 0.006 0.003)
entropy_coef_values=(0.03)
seeds=(1 2 3 4 5 6 7 8 9 10)

# Calculate indices for each parameter
combo_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))

# Calculate indices for each hyperparameter within the combo
lrConSpec_index=$((combo_index / ${#intrinsicR_scale_values[@]}))
intrinsicR_scale_index=$((combo_index % ${#intrinsicR_scale_values[@]}))

# Get parameter values
lr=${lr_values[0]}
intrinsicR_scale=${intrinsicR_scale_values[$intrinsicR_scale_index]}
lrConSpec=${lrConSpec_values[$lrConSpec_index]}
entropy_coef=${entropy_coef_values[0]}
seed=${seeds[$seed_index]}

# Run the command with the specified hyperparameters
python main.py --env-name MultiDoorKeyEnv-6x6-2keys-v0 \
    --use-linear-lr-decay \
    --save_checkpoint \
    --save-interval 5000 \
    --lr $lr \
    --num-processes 8 \
    --entropy-coef $entropy_coef \
    --num-mini-batch 4 \
    --num-epochs 20000 \
    --intrinsicR_scale $intrinsicR_scale \
    --lrConSpec $lrConSpec \
    --seed $seed
