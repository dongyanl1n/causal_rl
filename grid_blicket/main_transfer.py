import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_minigrid_envs
from a2c_ppo_acktr.model import Policy  # taken from ConSpec repo's a2c_ppo_acktr/modelRL.py
from a2c_ppo_acktr.storage import RolloutStorage  # taken from ConSpec repo's a2c_ppo_acktr/storage.py
from arguments import get_args
from Conspec.ConSpec import ConSpec
import datetime
import sys
import wandb

# @profile
def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.env_name == "MiniGrid-FourRooms-v0":
        train_envs, train_max_steps = make_minigrid_envs(
            num_envs=args.num_processes,
            env_name=args.env_name,
            seeds=[(args.seed + i) for i in range(args.num_processes)],
            device=device,
            goal_pos=(1, 1)
            )
        # Transfer environment (goal at (15,6))
        transfer_envs, transfer_max_steps = make_minigrid_envs(
            num_envs=args.num_processes,
            env_name=args.env_name,
            seeds=[(args.seed + args.num_processes + i) for i in range(args.num_processes)],
            device=device,
            goal_pos=(15, 6)
            )
    else:
        pass
    args.num_steps = train_max_steps
    obsspace = train_envs.observation_space
    actionspace = train_envs.action_space
    print('obsspace.shape', obsspace.shape)
    print('actionspace', actionspace)
    print('use_recurrent_policy', args.recurrent_policy)
    actor_critic = Policy(
        obsspace.shape,
        actionspace,
        base_kwargs={'recurrent': args.recurrent_policy,
                     'observation_space': obsspace})
    actor_critic.to(device)

    wandb.init(project="fourrooms", 
            entity="dongyanl1n", 
            name=f"{args.env_name}-conspec-rec-PO-lr{args.lr}-seed{args.seed}",
            dir="/network/scratch/l/lindongy/grid_blickets",
            tags=['Transfer'],
            config=args)


    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    ###############CONSPEC FUNCTIONS##############################
    '''
    Here, the main ConSpec class is loaded. All the relevant ConSpec functions and objects are contained in this class.
    '''
    conspecfunction = ConSpec(args, obsspace, actionspace, device)
    ##############################################################
    print('steps', args.num_steps)
    num_updates = 20000
    switch_point = 10000
    print('num_updates', num_updates)
    print('switching to transfer environment at update', switch_point)

    # Initialize two separate rollout storages
    train_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                    train_envs.observation_space.shape,
                                    actor_critic.recurrent_hidden_state_size)
    train_rollouts.to(device)
    transfer_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                    transfer_envs.observation_space.shape,
                                    actor_critic.recurrent_hidden_state_size)
    transfer_rollouts.to(device)

    # Start with train_envs
    current_envs = train_envs
    current_rollouts = train_rollouts

    obs = current_envs.reset()
    current_rollouts.obs[0].copy_(torch.transpose(obs, 3, 1))
    episode_rewards = deque(maxlen=int(args.num_processes*args.log_interval))
    episode_lengths = deque(maxlen=int(args.num_processes*args.log_interval))
    
    start = time.time()
    # env_frames = {i: [] for i in range(args.num_processes)}  # to make a video of training
    ext_rewards = []
    ext_int_rewards = []

    for j in range(num_updates):  # one rollout/episode per update
        # Switch to transfer environment at the switch point
        if j == switch_point:
            print("Switching to transfer environment")
            current_envs = transfer_envs
            current_rollouts = transfer_rollouts

        obs = current_envs.reset()
        current_rollouts.obs[0].copy_(torch.transpose(obs, 3, 1))
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    current_rollouts.obs[step], current_rollouts.recurrent_hidden_states[step],
                    current_rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = current_envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_lengths.append(info['episode']['l'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = masks
            current_rollouts.insert(torch.transpose(obs, 3, 1), recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

            # for i in range(args.num_processes):
            #     vobs = obs[i].cpu().detach().numpy()
            #     vobs = (vobs * 255).astype(np.uint8)
            #     env_frames[i].append(vobs) # to make a video of training


        ########### episode finishes ############

        with torch.no_grad():
            next_value = actor_critic.get_value(
                current_rollouts.obs[-1], current_rollouts.recurrent_hidden_states[-1],
                current_rollouts.masks[-1]).detach()
            # now compute new rewards
            rewardstotal = current_rollouts.retrieveR()  # gets extrinsic reward, i.e. rollouts.rewards  # torch.Size([ep_length, num_processes, 1])
            ext_rewards.append(rewardstotal.sum(0).mean().cpu().detach().numpy())  # sum over time, mean over processes

        ###############CONSPEC FUNCTIONS##############################
        '''
        The purpose here is to: 
        1. retrieve the current minibatch of trajectory (including its observations, rewards, hidden states, actions, masks)
        2. "do everything" that ConSpec needs to do internally for training, and output the intrinsic + extrinsic reward for the current minibatch of trajectories
        3. store this total reward in the memory buffer 
        '''
        obstotal, rewardtotal, recurrent_hidden_statestotal, actiontotal,  maskstotal  = current_rollouts.release()

        reward_intrinsic_extrinsic  = conspecfunction.do_everything(obstotal, recurrent_hidden_statestotal, actiontotal, rewardtotal, maskstotal)
        ext_int_rewards.append(reward_intrinsic_extrinsic.sum(0).mean().cpu().detach().numpy())  # sum over time, mean over processes
        current_rollouts.storereward(reward_intrinsic_extrinsic)  # update rollouts.rewards to be reward_intrinsic_extrinsic, i.e. the sum of intrinsic and extrinsic rewards
        ##############################################################
        
        current_rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)  # compute returns using updated rewards (intrinsic+extrinsic)

        value_loss, action_loss, dist_entropy = agent.update(current_rollouts)  # update actor_critic based on returns and value_preds

        current_rollouts.after_update()  # last step becomes first step of next rollout

        ############# Log and save #################
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            # calculate moving average of ext_rewards and ext_int_rewards to calculate the moving average of intrinsic rewards
            int_rewards_ema = np.mean(ext_int_rewards[-args.num_processes*args.log_interval:]) - np.mean(ext_rewards[-args.num_processes*args.log_interval:])
            print(f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}")
            print(f'Moving average for external rewards: {np.mean(episode_rewards):.5f}, for episode length {np.mean(episode_lengths):.5f}, for intrinsic rewards: {int_rewards_ema:.1f}')
            wandb.log({
                'Epoch': j,
                'total_num_steps': total_num_steps,
                'FPS': int(total_num_steps / (end - start)),
                'mean_reward': np.mean(episode_rewards),
                # 'median_reward': np.median(episode_rewards),
                # 'min_reward': np.min(episode_rewards),
                # 'max_reward': np.max(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                # 'median_length': np.median(episode_lengths),
                # 'min_length': np.min(episode_lengths),
                # 'max_length': np.max(episode_lengths),
                'dist_entropy': dist_entropy,
                'value_loss': value_loss,
                'action_loss': action_loss,
                'intrinsic_reward': int_rewards_ema
            })

    wandb.finish()
    train_envs.close()
    transfer_envs.close()


if __name__ == "__main__":
    main()