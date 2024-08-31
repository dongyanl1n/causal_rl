#!/bin/bash
#SBATCH --job-name=wandb-sweep-8x8-2keys
#SBATCH --output=/network/scratch/l/lindongy/grid_blickets/sbatch_log/wandb-sweep-8x8-2keys_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/grid_blickets/sbatch_log/wandb-sweep-8x8-2keys_%A_%a.err
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --array=1-64

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

SWEEP_ID="46e7os84"

python sweep_agent.py $SWEEP_ID