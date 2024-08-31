#!/bin/bash
#SBATCH --job-name=putnext
#SBATCH --output=/network/scratch/l/lindongy/blicket_objects_env/sbatch_log/putnext_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/blicket_objects_env/sbatch_log/putnext_%A_%a.err
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=20:00:00
#SBATCH --array=0-15


module load python/3.7
source $HOME/envs/causal_rl/bin/activate

# Define parameter arrays
lr_values=(0.0002 0.0007)
intrinsicR_scale_values=(0.2 0.5)
lrConSpec_values=(0.002 0.005)
seeds=(1 2)

# Calculate indices for each parameter
combo_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))

lr_index=$((combo_index / (${#intrinsicR_scale_values[@]} * ${#lrConSpec_values[@]})))
intrinsicR_scale_index=$(((combo_index / ${#lrConSpec_values[@]}) % ${#intrinsicR_scale_values[@]}))
lrConSpec_index=$((combo_index % ${#lrConSpec_values[@]}))

# Get parameter values
lr=${lr_values[$lr_index]}
intrinsicR_scale=${intrinsicR_scale_values[$intrinsicR_scale_index]}
lrConSpec=${lrConSpec_values[$lrConSpec_index]}
seed=${seeds[$seed_index]}

# Run the command with the specified hyperparameters
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python main.py --env-name MiniWorld-PutNext-v0 \
    --use-linear-lr-decay \
    --lr $lr \
    --num-processes 4 \
    --entropy-coef 0.02 \
    --num-mini-batch 4 \
    --num-epochs 10000 \
    --intrinsicR_scale $intrinsicR_scale \
    --lrConSpec $lrConSpec \
    --save_checkpoint \
    --seed $seed
    

