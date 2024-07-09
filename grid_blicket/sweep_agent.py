import wandb
import subprocess

def train():
    # Initialize wandb
    run = wandb.init(reinit=True, entity="dongyanl1n", project="grid_blicket_env", dir="/network/scratch/l/lindongy/grid_blickets")
    
    # Access config values
    config = wandb.config

    env_name = "MultiDoorKeyEnv-8x8-2keys-v0"
    
    cmd = [
        "python", "-u", "main.py", "--use-linear-lr-decay",
        "--env-name", env_name,
        "--lr", str(config.lr),
        "--intrinsicR_scale", str(config.intrinsicR_scale),
        "--lrConSpec", str(config.lrConSpec),
        "--entropy-coef", str(config.entropy_coef),
        "--num-mini-batch", str(config.num_mini_batch)
    ]
    
    subprocess.call(cmd)

if __name__ == "__main__":
    sweep_id = "46e7os84"
    wandb.agent(sweep_id, function=train, entity="dongyanl1n", project="grid_blicket_env")