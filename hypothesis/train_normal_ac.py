import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random

from args import get_args
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_minigrid_envs
from a2c_ppo_acktr.model_old import Policy  # normal policy network, not taking hypothesis as input
from a2c_ppo_acktr.storage import RolloutStorage  # taken from ConSpec repo's a2c_ppo_acktr/storage.py
from Conspec.ConSpec import ConSpec
import datetime
import wandb


def load_partial_weights(actor_critic, checkpoint):
    """
    Load useful components of pretrained non-goal conditioned actor_critic model.
    """
    state_dict = checkpoint['actor_critic_state_dict']
    model_dict = actor_critic.state_dict()

    # # Load GRU weights
    # for key in ['base.gru.weight_ih_l0', 'base.gru.weight_hh_l0', 'base.gru.bias_ih_l0', 'base.gru.bias_hh_l0']:
    #     model_dict[key] = state_dict[key]

    # Load CNN weights
    model_dict['base.cnn.0.weight'] = state_dict['base.cnn.0.weight']
    model_dict['base.cnn.0.bias'] = state_dict['base.cnn.0.bias']

    # # Load critic weights
    # model_dict['base.critic_linear.weight'] = state_dict['base.critic_linear.weight']
    # model_dict['base.critic_linear.bias'] = state_dict['base.critic_linear.bias']

    # # Load actor weights
    # model_dict['dist.linear.weight'] = state_dict['dist.linear.weight']
    # model_dict['dist.linear.bias'] = state_dict['dist.linear.bias']

    # Load the updated state dict
    actor_critic.load_state_dict(model_dict, strict=False)


# @profile
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #========================= for single wandb job ==============================
    #wandb.init(project="hypothesis", 
    #        entity="dongyanl1n", 
    #        name=f"{args.env_name}-conspec-rec-PO-lr{args.lr}-intrinsicR_scale{args.intrinsicR_scale}-seed{args.seed}",
    #        dir=os.environ.get('SLURM_TMPDIR', '/network/scratch/l/lindongy/hypothesis'),
    #        tags=['load_frozenSF_only', '11111111'],
    #        config=args)
    #========================================================================
    
    #========================= for single comet job ==============================
    # Initialize Comet ML experiment
    # experiment = Experiment(
    #     api_key="QeWdbZ4T3xigB5rCZuzGWzh2G",
    #     project_name="hypothesis",
    #     workspace="dongyanl1n"
    # )

    # # Set the experiment name
    # experiment.set_name(f"{args.env_name}-conspec-rec-PO-lr{args.lr}-intrinsicR_scale{args.intrinsicR_scale}-seed{args.seed}")
    
    # # Set the experiment tags
    # experiment.add_tags(['load_frozenSF_only', '11111111'])
    
    # # Log the configuration parameters
    # experiment.log_parameters(vars(args))
    
    # # Set the output directory
    # output_dir = os.environ.get('SLURM_TMPDIR', '/network/scratch/l/lindongy/hypothesis')
    # experiment.log_other('output_dir', output_dir)
    #========================================================================

    # create folder for checkpoints
    if args.save_checkpoint:
        base_directory = "/network/scratch/l/lindongy/hypothesis/conspec_ckpt"
        subfolder_name = f"{args.env_name}-conspec-rec-PO-lr{args.lr}-intrinsicR_scale{args.intrinsicR_scale}-seed{args.seed}"
        full_path = os.path.join(base_directory, subfolder_name)
        os.makedirs(full_path, exist_ok=True)
    
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs, max_steps = make_minigrid_envs(
        num_envs=args.num_processes,
        env_name=args.env_name,
        device=device,
        fixed_positions=args.fixed_positions,
        no_ret_normalization=True
        )
    args.num_steps = max_steps
    obsspace = envs.observation_space
    actionspace = envs.action_space
    print('obsspace.shape', obsspace.shape)
    print('actionspace', actionspace)
    print('use_recurrent_policy', args.recurrent_policy)

    # hypothesis = torch.zeros(1, args.num_prototypes).to(device)
    # hypothesis[0, 0::2] = 1
    hypothesis = torch.ones(1, args.num_prototypes).to(device)
    print('hypothesis', hypothesis)
    if not 0 in hypothesis:
        print("Using all prototypes. Updating agent with Rit+Ret")
    else:
        print("Updating agent with Rit")

    actor_critic = Policy(
        obsspace.shape,
        actionspace,
        base_kwargs={'recurrent': args.recurrent_policy,
                     'observation_space': obsspace})
    actor_critic.to(device)

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

    ############# Load pretrained model and frozen SF buffer ################
    # Load checkpoints of trained agent
    base_path = args.base_path
    ckpt_epi = args.ckpt_epi
    checkpoint_path = os.path.join(base_path, f"model_checkpoint_epoch_{ckpt_epi}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    conspecfunction.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    conspecfunction.prototypes.prototypes.load_state_dict(checkpoint['prototypes'])
    conspecfunction.prototypes.layers1.load_state_dict(checkpoint['layers1'])
    conspecfunction.prototypes.layers2.load_state_dict(checkpoint['layers2'])
    load_partial_weights(actor_critic, checkpoint)
    print(f"Loaded model from {checkpoint_path}")

    # load frozen SF buffers
    frozen_buffer_path = os.path.join(base_path, f"conspec_rollouts_frozen_epoch_{ckpt_epi}.pth")
    conspecfunction.rollouts.load_SFbuffer_frozen(frozen_buffer_path)
    print(f"Loaded frozen buffer from {frozen_buffer_path}")

    # freeze conspec
    conspecfunction.freeze_parameters()
    ##############################################################

    print('steps', args.num_steps)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obsspace.shape, 
                              actor_critic.recurrent_hidden_state_size)
    rollouts.to(device)
    obs = envs.reset()
    rollouts.obs[0].copy_(torch.transpose(obs, 3, 1))
    episode_rewards = deque(maxlen=int(args.num_processes*args.log_interval))
    episode_lengths = deque(maxlen=int(args.num_processes*args.log_interval))
    
    # num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes  # number of training episodes per environment
    num_updates = args.num_epochs
    print('num_updates', num_updates)
    start = time.time()
    # env_frames = {i: [] for i in range(args.num_processes)}  # to make a video of training
    ext_rewards = []
    ext_int_rewards = []

    for j in range(num_updates):  # one rollout/episode per update
        obs = envs.reset()
        rollouts.obs[0].copy_(torch.transpose(obs, 3, 1))
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_lengths.append(info['episode']['l'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = masks
            rollouts.insert(torch.transpose(obs, 3, 1), recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

            # for i in range(args.num_processes):
            #     vobs = obs[i].cpu().detach().numpy()
            #     vobs = (vobs * 255).astype(np.uint8)
            #     env_frames[i].append(vobs) # to make a video of training


        ########### episode finishes ############

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
            # now compute new rewards
            rewardstotal = rollouts.retrieveR()  # gets extrinsic reward, i.e. rollouts.rewards  # torch.Size([ep_length, num_processes, 1])
            ext_rewards.append(rewardstotal.sum(0).mean().cpu().detach().numpy())  # sum over time, mean over processes

        ###############CONSPEC FUNCTIONS##############################
        '''
        The purpose here is to: 
        1. retrieve the current minibatch of trajectory (including its observations, rewards, hidden states, actions, masks)
        2. "do everything" that ConSpec needs to do internally for training, and output the intrinsic + extrinsic reward for the current minibatch of trajectories
        3. store this total reward in the memory buffer 
        '''
        obstotal, rewardtotal, recurrent_hidden_statestotal, actiontotal,  maskstotal  = rollouts.release()

        with torch.no_grad():
            conspecfunction.store_memories(obstotal, recurrent_hidden_statestotal, actiontotal, rewardtotal, maskstotal)  # store in main buffer only
            reward_intrinsic_extrinsic, reward_intrinsic = conspecfunction.calc_intrinsic_reward_v0(prototypes_used=hypothesis)

        ext_int_rewards.append(reward_intrinsic_extrinsic.sum(0).mean().cpu().detach().numpy())  # sum over time, mean over processes
        if not 0 in hypothesis:  # setting all prototypes to True; means that we care about external reward
            rollouts.storereward(reward_intrinsic_extrinsic)  # update rollouts.rewards to be reward_intrinsic_extrinsic, i.e. the sum of intrinsic and extrinsic rewards
        else:
            rollouts.storereward(reward_intrinsic.unsqueeze(-1))  # update rollouts.rewards to be reward_intrinsic, i.e. the sum of intrinsic and extrinsic rewards
        ##############################################################
        
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)  # compute returns using updated rewards (intrinsic+extrinsic)

        # value_loss, action_loss, dist_entropy = agent.update(rollouts, hypothesis_batch)  # update actor_critic based on returns and value_preds
        value_loss, action_loss, dist_entropy = agent.update(rollouts)  # update actor_critic based on returns and value_preds
        rollouts.after_update()  # last step becomes first step of next rollout

        ############# Log and save #################
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            # calculate moving average of ext_rewards and ext_int_rewards to calculate the moving average of intrinsic rewards
            int_rewards_ema = np.mean(ext_int_rewards[-args.num_processes*args.log_interval:]) - np.mean(ext_rewards[-args.num_processes*args.log_interval:])
            print(f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}")
            print(f'Moving average for episode rewards: {np.mean(episode_rewards):.5f}, for episode length {np.mean(episode_lengths):.5f}')
            print(f"for external reward: {np.mean(ext_rewards[-args.num_processes*args.log_interval:]):5f}, for ext+int reward: {np.mean(ext_int_rewards[-args.num_processes*args.log_interval:]):5f}, for intrinsic rewards: {int_rewards_ema:.5f}")
                  
            #wandb.log({
            #    'Epoch': j,
            #    'total_num_steps': total_num_steps,
            #    'FPS': int(total_num_steps / (end - start)),
            #    'mean_reward': np.mean(episode_rewards),
            #    # 'median_reward': np.median(episode_rewards),
            #    # 'min_reward': np.min(episode_rewards),
            #    # 'max_reward': np.max(episode_rewards),
            #    'mean_length': np.mean(episode_lengths),
            #    # 'median_length': np.median(episode_lengths),
            #    # 'min_length': np.min(episode_lengths),
            #    # 'max_length': np.max(episode_lengths),
            #    'dist_entropy': dist_entropy,
            #    'value_loss': value_loss,
            #    'action_loss': action_loss,
            #    'intrinsic_reward': int_rewards_ema,
            #})
    # wandb.finish()
    #         experiment.log_metrics({
    #             'Epoch': j,
    #             'total_num_steps': total_num_steps,
    #             'FPS': int(total_num_steps / (end - start)),
    #             'mean_reward': np.mean(episode_rewards),
    #             'mean_length': np.mean(episode_lengths),
    #             'dist_entropy': dist_entropy,
    #             'value_loss': value_loss,
    #             'action_loss': action_loss,
    #             'intrinsic_reward': int_rewards_ema,
    #         }, step=total_num_steps)
    # experiment.end()


if __name__ == "__main__":
    main()
