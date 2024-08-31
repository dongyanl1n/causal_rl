# Grid Blicket

This repo is named as such because the goal is to train conspec on MiniGrid-based, Blicket-decector-inspired environment. One such environment is MultiDoorKeyEnv in multidoorkey_env.py, where one can specify the size of the environment and the number of key-door pairs needed to be opened before agent can proceed to the goal. 

The agents used in this repo are ConSpec (main.py), Ilya Kostrikov's PPO (baseline.py), and Stable Baselines 3's PPO (minigrid_sb3.py). 

## Usage

Both main.py and baseline.py receives arguments from arguments.py.

To run ConSpec on MultiDoorKeyEnv, one could do
```
python main.py --env-name MultiDoorKeyEnv-6x6-2keys-v0 \
    --use-linear-lr-decay \
    --lr 0.0006 \
    --num-processes 8 \
    --entropy-coef 0.02 \
    --num-mini-batch 4 \
    --intrinsicR_scale 0.1 \
    --lrConSpec 0.01
```
With all other hyperparameters being set to default values, this would be the best hyperparameter setting that gives stable performance across seeds.

To run a grid search for best hyperparameters, one could do with an array job, with
```
sbatch submit_grid_search.sh
```

## Misc
- Anything that has the word "sweep" in it was my attempt to do hyperparameter sweep with wandb, which in my experience wasn't as efficient as just a simple grid search. Still Work In Progress (WIP).
- embedding.py and analysis_utils.py are my draft scripts for analysing prototypes, etc. Still Work In Progress (WIP).