#!/bin/bash
#SBATCH --job-name=hypothesis_all1
#SBATCH --output=/network/scratch/l/lindongy/hypothesis/sbatch_log/hypothesis_all1_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/hypothesis/sbatch_log/hypothesis_all1_%A_%a.err
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=12:00:00
#SBATCH --array=0-7

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

# Define parameter arrays
lr_values=(0.0006)
intrinsicR_scale_values=(0.05)
cos_score_threshold_values=(0.99)
entropy_coef_values=(0.02)
seeds=(4 5 6 7 8 9 10 11)

# Calculate indices for each parameter
combo_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))

entropy_coef_index=$((combo_index / (${#intrinsicR_scale_values[@]} * ${#cos_score_threshold_values[@]})))
intrinsicR_scale_index=$(((combo_index / ${#cos_score_threshold_values[@]}) % ${#intrinsicR_scale_values[@]}))
cos_score_threshold_index=$((combo_index % ${#cos_score_threshold_values[@]}))

# Get parameter values
lr=${lr_values[0]}
intrinsicR_scale=${intrinsicR_scale_values[$intrinsicR_scale_index]}
cos_score_threshold=${cos_score_threshold_values[$cos_score_threshold_index]}
entropy_coef=${entropy_coef_values[$entropy_coef_index]}
seed=${seeds[$seed_index]}

# Run the command with the specified hyperparameters
python train_normal_ac.py --env-name MultiDoorKeyEnv-6x6-2keys-v0 \
    --use-linear-lr-decay \
    --lr $lr \
    --num-processes 8 \
    --entropy-coef $entropy_coef \
    --num-mini-batch 4 \
    --num-epochs 10000 \
    --intrinsicR_scale $intrinsicR_scale \
    --seed $seed \
    --cos_score_threshold $cos_score_threshold
