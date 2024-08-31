#!/bin/bash
#SBATCH --job-name=sb3-hallway
#SBATCH --output=/network/scratch/l/lindongy/blicket_objects_env/sbatch_log/sb3-hallway_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/blicket_objects_env/sbatch_log/sb3-hallway_%A_%a.err
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --array=1-1

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

# Define hyperparameters
learning_rates=(0.0001)

xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python miniworld_sb3.py \
    --lr ${learning_rates[$SLURM_ARRAY_TASK_ID-1]} \
    --total_timesteps 200000 \
    --env_name MiniWorld-Hallway-v0