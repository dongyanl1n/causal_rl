#!/bin/bash
#SBATCH --job-name=dt-ppomixture
#SBATCH --output=/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/dt-ppomixture_%A_%a.out
#SBATCH --error=/network/scratch/l/lindongy/causal_overhypotheses/sbatch_log/dt-ppomixture_%A_%a.err
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec
#SBATCH --array=0-4

module load python/3.8
source ~/envs/decision-transformer-gym/bin/activate

# Define an array of commands
commands=(
    'python experiment.py --env causal --dataset ppomixture --holdout_strategy none --K 40 --model dt --batch_size 128 --num_eval_episodes 100 --log_to_wandb True'
    'python experiment.py --env causal --dataset ppoexpert --holdout_strategy none --K 40 --model dt --batch_size 128 --num_eval_episodes 100 --log_to_wandb True'
    'python experiment.py --env causal --dataset random30steps --holdout_strategy none --K 40 --model dt --batch_size 128 --num_eval_episodes 100 --log_to_wandb True --n_layer 6 --n_head 8'
    'python experiment.py --env causal --dataset ppomixture --holdout_strategy none --K 40 --model dt --batch_size 128 --num_eval_episodes 100 --log_to_wandb True --n_layer 6 --n_head 8'
    'python experiment.py --env causal --dataset ppoexpert --holdout_strategy none --K 40 --model dt --batch_size 128 --num_eval_episodes 100 --log_to_wandb True --n_layer 6 --n_head 8'
)

# Execute the command corresponding to this task
${commands[$SLURM_ARRAY_TASK_ID]}