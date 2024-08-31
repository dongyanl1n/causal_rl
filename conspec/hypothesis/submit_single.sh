#!/bin/bash
#SBATCH --job-name=hypothesis
#SBATCH --output=/network/scratch/l/lindongy/hypothesis/sbatch_log/hypothesis_%j.out
#SBATCH --error=/network/scratch/l/lindongy/hypothesis/sbatch_log/hypothesis_%j.err
#SBATCH --partition=main
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=12:00:00

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

python train_normal_ac.py --env-name MultiDoorKeyEnv-6x6-2keys-v0 --use-linear-lr-decay  --lr 0.0006 --num-processes 8 --entropy-coef 0.02 --num-mini-batch 4 --num-epochs 10000