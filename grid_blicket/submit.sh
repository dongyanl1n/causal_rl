#!/bin/bash
#SBATCH --job-name=emptyenv
#SBATCH --output=/network/scratch/l/lindongy/emptyenv/sbatch_log/%j_%a.out
#SBATCH --error=/network/scratch/l/lindongy/emptyenv/sbatch_log/%j_%a.err
#SBATCH --partition=long
#SBATCH --mem=4G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --array=1-4

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

case $SLURM_ARRAY_TASK_ID in
  1)
    python emptyenv.py --lr 1e-4 --total_timesteps 2e6 --env MiniGrid-Empty-Random-6x6-v0
    ;;
  2)
    python emptyenv.py --lr 5e-5 --total_timesteps 2e6 --env MiniGrid-Empty-Random-6x6-v0
    ;;
  3)
    python emptyenv.py --lr 1e-4 --total_timesteps 2e6 --env MiniGrid-Empty-16x16-v0
    ;;
  4)
    python emptyenv.py --lr 5e-5 --total_timesteps 2e6 --env MiniGrid-Empty-16x16-v0
    ;;
esac
