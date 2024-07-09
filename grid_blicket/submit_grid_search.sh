#!/bin/bash
#SBATCH --job-name=6x6-2keys
#SBATCH --output=/network/scratch/l/lindongy/grid_blickets/sbatch_log/6x6-2keys_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/grid_blickets/sbatch_log/6x6-2keys_%A_%a.err
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --array=0-191

# This will run 192 different combinations of hyperparameters

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

# Define parameter arrays
lr_values=(0.0001 0.0003 0.0006 0.001)
intrinsicR_scale_values=(0.2 0.3 0.4 0.5)
lrConSpec_values=(0.001 0.003 0.006 0.01)
entropy_coef_values=(0.03 0.04 0.05)

# Calculate indices for each parameter
lr_index=$((SLURM_ARRAY_TASK_ID / (${#intrinsicR_scale_values[@]} * ${#lrConSpec_values[@]} * ${#entropy_coef_values[@]})))
intrinsicR_scale_index=$(((SLURM_ARRAY_TASK_ID / (${#lrConSpec_values[@]} * ${#entropy_coef_values[@]})) % ${#intrinsicR_scale_values[@]}))
lrConSpec_index=$(((SLURM_ARRAY_TASK_ID / ${#entropy_coef_values[@]}) % ${#lrConSpec_values[@]}))
entropy_coef_index=$((SLURM_ARRAY_TASK_ID % ${#entropy_coef_values[@]}))

# Get parameter values
lr=${lr_values[$lr_index]}
intrinsicR_scale=${intrinsicR_scale_values[$intrinsicR_scale_index]}
lrConSpec=${lrConSpec_values[$lrConSpec_index]}
entropy_coef=${entropy_coef_values[$entropy_coef_index]}

# Run the command with the specified hyperparameters
python main.py --env-name MultiDoorKeyEnv-6x6-2keys-v0 \
    --use-linear-lr-decay \
    --lr $lr \
    --num-processes 8 \
    --entropy-coef $entropy_coef \
    --num-mini-batch 4 \
    --num-epochs 20000 \
    --use-linear-lr-decay \
    --intrinsicR_scale $intrinsicR_scale \
    --lrConSpec $lrConSpec

cp -r $SLURM_TMPDIR/wandb /network/scratch/l/lindongy/grid_blickets/
