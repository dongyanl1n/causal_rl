#!/bin/bash
#SBATCH --job-name=putnexteasy
#SBATCH --output=/network/scratch/l/lindongy/blicket_objects_env/sbatch_log/putnexteasy_%j.out
#SBATCH --error=/network/scratch/l/lindongy/blicket_objects_env/sbatch_log/putnexteasy_%j.err
#SBATCH --partition=long
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1

module load python/3.7
source $HOME/envs/causal_rl/bin/activate

xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python main.py --env-name MiniWorld-PutNext-v0 \
    --use-linear-lr-decay \
    --lr 0.0007 \
    --num-processes 4 \
    --entropy-coef 0.02 \
    --num-mini-batch 4 \
    --num-epochs 2000 \
    --intrinsicR_scale 0.5 \
    --lrConSpec 0.002 \
    --save_checkpoint \
    --seed 1