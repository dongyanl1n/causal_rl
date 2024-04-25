
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../envs'))
import argparse
from stable_baselines import A2C, PPO2
from causal_env_v0 import CausalEnv_v0
import tqdm
import numpy as np
import pickle

def main(args):
    print('Loading model...')
    # Load the model
    model_path = "/network/scratch/l/lindongy/causal_overhypotheses/model_output/{}/model".format(args.model_name)
    if args.model_name is None:
        model = None
    elif 'a2c' in args.model_name:
        model = A2C.load(model_path)
    elif 'ppo2' in args.model_name:
        model = PPO2.load(model_path)
    else:
        raise ValueError('Unknown model type {}'.format(args.model_name))

    # Create an environment
    env = CausalEnv_v0({
        "reward_structure":  args.reward_structure,
        "quiz_disabled_steps": args.quiz_disabled_steps,
    })

    # Roll out the environment for k trajectories
    print('Collecting Trajectories...')
    trajectories = []
    for i in tqdm.tqdm(range(args.num_trajectories)):
        # Reset the environment
        obs = env.reset()
        gt = str(env._current_gt_hypothesis).split("'")[1].split('.')[-1]

        # Roll out the environment for n steps
        steps = []
        for j in range(args.max_steps):
            # Get the action from the model
            if model is not None:
                # Because the environment is stacked, we have to extract x2
                action = model.predict(np.stack([obs, obs, obs, obs]), deterministic=True)[0][0]
            else:  # Random action
                action = env.action_space.sample()


            # Step the environment
            n_obs, reward, done, info = env.step(action)

            steps.append((obs, action, reward, n_obs, done))
            obs = n_obs

            # Check if the episode has ended
            if done:
                break

        observations = np.stack([step[0] for step in steps])
        next_obeservations = np.stack([step[3] for step in steps])
        actions = np.stack([step[1] for step in steps])
        rewards = np.stack([step[2] for step in steps])
        terminals = np.stack([step[4] for step in steps])
        trajectories.append({
            'gt': gt,
            'observations': observations,
            'next_observations': next_obeservations,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals
        })

    # Save the trajectories
    print('Saving Trajectories...')
    if args.model_name is None:
        save_name = "random_action"
        save_name += ('_qd=' + str(args.quiz_disabled_steps)) if args.quiz_disabled_steps > 0 else ''
        save_name += ('_rs=' + str(args.reward_structure))
        os.makedirs('/network/scratch/l/lindongy/causal_overhypotheses/model_output/{}'.format(save_name), exist_ok=True)
        output_path = "/network/scratch/l/lindongy/causal_overhypotheses/model_output/{}/trajectories.pkl".format(save_name)
    else:
        output_path = "/network/scratch/l/lindongy/causal_overhypotheses/model_output/{}/trajectories.pkl".format(args.model_name)
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Trajectories from Causal Environments')
    parser.add_argument('--env', type=str, default='CausalEnv_v0', help='Environment to use')
    parser.add_argument('--model_name', type=str, default=None, help='save_name in driver.py')
    parser.add_argument('--num_trajectories', type=int, default=10000, help='Number of trajectories to collect')
    parser.add_argument('--max_steps', type=int, default=30, help='Maximum number of steps per trajectory')
    parser.add_argument('--quiz_disabled_steps', type=int, default=-1, help='Number of steps to disable quiz')
    parser.add_argument('--reward_structure', type=str, default='baseline', help='Reward structure')
    args = parser.parse_args()
    print(args)
    main(args)
