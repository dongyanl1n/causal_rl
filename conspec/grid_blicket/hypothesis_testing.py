'''
Train all-1-conditioned policy with ConSpec on task.
'''
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_minigrid_envs
from a2c_ppo_acktr.model_hypothesis import Hypothesis_Policy
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
    np.random.seed(args.seed)
    random.seed(args.seed)
    checkpoint_saved = False
    good_performance_counter = 0
    #========================= for wandb sweep ==============================
    # Initialize wandb first
    # wandb.init(project="grid_blicket_env", 
    #     entity="dongyanl1n",
    #     dir="/network/scratch/l/lindongy/grid_blickets",
    #     config=args.__dict__)  # Pass all arguments as initial config

    # # Now update args with wandb.config values
    # args.lr = wandb.config.lr
    # args.intrinsicR_scale = wandb.config.intrinsicR_scale
    # args.lrConSpec = wandb.config.lrConSpec
    # args.entropy_coef = wandb.config.entropy_coef
    # args.num_mini_batch = wandb.config.num_mini_batch

    # # Update wandb run name after getting the final config values
    # wandb.run.name = f"{args.env_name}-conspec-rec-PO-lr{args.lr}-seed{args.seed}"
    # wandb.run.save()
    #========================================================================

    #========================= for single wandb job ==============================
    wandb.init(project="hypothesis_policy", 
            entity="dongyanl1n", 
            name=f"{args.env_name}-conspec-rec-{'FO' if args.fully_observed else 'PO'}-{args.hypothesis}-lr{args.lr}-intrinsR{args.intrinsicR_scale}-lrConSpec{args.lrConSpec}-entropy{args.entropy_coef}-seed{args.seed}",
            tags=[args.hypothesis],
            dir=os.environ.get('SLURM_TMPDIR', '/scratch/lindongy/hypothesis_policy'),
            config=args)
    #========================================================================

    # create folder for checkpoints
    if args.save_checkpoint:
        base_directory = "/scratch/lindongy/hypothesis_policy/conspec_ckpt"
        subfolder_name = f"{args.env_name}-conspec-rec-{'FO' if args.fully_observed else 'PO'}-{args.hypothesis}-lr{args.lr}-intrinsR{args.intrinsicR_scale}-lrConSpec{args.lrConSpec}-entropy{args.entropy_coef}-seed{args.seed}"
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
        fully_observed=args.fully_observed,
        max_steps=args.max_steps,
        no_ret_normalization=True
        )
    args.num_steps = max_steps
    obsspace = envs.observation_space
    actionspace = envs.action_space
    print(f"using {'full observability' if args.fully_observed else 'partial observability'}")
    print('obsspace.shape', obsspace.shape)
    print('actionspace', actionspace)
    print('use_recurrent_policy', args.recurrent_policy)
    assert len(args.hypothesis) == args.num_prototypes
    # turn hypothesis string into a tensor
    print('args.hypothesis: ', args.hypothesis)
    hypothesis = torch.cat([torch.tensor([int(i) for i in h]) for h in args.hypothesis]).to(device)
    hypothesis_batch = hypothesis.repeat(args.num_processes, 1)
    actor_critic = Hypothesis_Policy(
        obsspace.shape,
        actionspace,
        args.num_prototypes, 
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
    print('steps', max_steps)
    rollouts = RolloutStorage(max_steps, args.num_processes,
                              obsspace.shape, 
                              actor_critic.recurrent_hidden_state_size)
    rollouts.to(device)
    obs = envs.reset()
    rollouts.obs[0].copy_(torch.transpose(obs, 3, 1))
    episode_rewards = deque(maxlen=int(args.num_processes*args.log_interval))
    episode_lengths = deque(maxlen=int(args.num_processes*args.log_interval))
    loss_conspec_list = deque(maxlen=int(args.num_processes*args.log_interval))
    int_rewards = deque(maxlen=int(args.num_processes*args.log_interval))
    
    # num_updates = int(args.num_env_steps) // max_steps // args.num_processes  # number of training episodes per environment
    num_updates = args.num_epochs
    print('num_updates', num_updates)
    start = time.time()
    # env_frames = {i: [] for i in range(args.num_processes)}  # to make a video of training

    for j in range(num_updates):  # one rollout/episode per update
        obs = envs.reset()
        rollouts.obs[0].copy_(torch.transpose(obs, 3, 1))
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(max_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], hypothesis_batch, rollouts.recurrent_hidden_states[step],
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
                rollouts.obs[-1], hypothesis_batch, rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
            # now compute new rewards
            rewardstotal = rollouts.retrieveR()  # gets extrinsic reward, i.e. rollouts.rewards  # torch.Size([ep_length, num_processes, 1])

        ###############CONSPEC FUNCTIONS##############################
        '''
        The purpose here is to: 
        1. retrieve the current minibatch of trajectory (including its observations, rewards, hidden states, actions, masks)
        2. "do everything" that ConSpec needs to do internally for training, and output the intrinsic + extrinsic reward for the current minibatch of trajectories
        3. store this total reward in the memory buffer 
        '''
        obstotal, rewardtotal, recurrent_hidden_statestotal, actiontotal,  maskstotal  = rollouts.release()

        reward_intrinsic_extrinsic, reward_intrinsic, loss_conspec  = conspecfunction.do_everything(obstotal, recurrent_hidden_statestotal, actiontotal, rewardtotal, maskstotal)
        if type(loss_conspec) != int:  # loss_conspec is 0 if there's no success trajectory, and conspec does not get updated
            loss_conspec_list.append(loss_conspec.cpu().detach().numpy())
        rollouts.storereward(reward_intrinsic_extrinsic)  # update rollouts.rewards to be reward_intrinsic_extrinsic, i.e. the sum of intrinsic and extrinsic rewards
        
        # log rewards
        reward_intrinsic = reward_intrinsic.sum(0).cpu().detach().squeeze()
        for i in range(len(reward_intrinsic)):
            int_rewards.append(reward_intrinsic[i].item())
        ##############################################################
        
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)  # compute returns using updated rewards (intrinsic+extrinsic)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, hypothesis_batch)  # update actor_critic based on returns and value_preds

        rollouts.after_update()  # last step becomes first step of next rollout
        if np.mean(episode_rewards)> 9.5:  # must consecutively have good performance for 100 epochs before saving checkpoint
            good_performance_counter += 1
        else:
            good_performance_counter = 0
        ############# Log and save #################
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            total_num_steps = (j + 1) * args.num_processes * max_steps
            end = time.time()
            print(f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}")
            print(f'Moving average for external rewards: {np.mean(episode_rewards):.5f}, for episode length {np.mean(episode_lengths):.5f}, for intrinsic rewards: {np.mean(int_rewards):.5f}')
            print(f'Losses: value {value_loss:.5f}, action {action_loss:.5f}, entropy {dist_entropy:.5f}, ConSpec {np.mean(loss_conspec_list):.5f}')
            print(f'ConSpec prototypes used: {conspecfunction.rollouts.prototypesUsed}, timestep count: {conspecfunction.rollouts.count_prototypes_timesteps_criterion}')
            
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
                'intrinsic_reward': np.mean(int_rewards),
                'loss_conspec': np.mean(loss_conspec_list),
                'good_performance_counter': good_performance_counter,
            })
        
        # save checkpoint
        if args.save_checkpoint:
            if ((j+1) % args.save_interval == 0 or j == num_updates - 1) and not checkpoint_saved and good_performance_counter >= 100:
                buffer = {
                    'obs': rollouts.obs,
                    'rewards': rollouts.rewards,
                    'hidden_states': rollouts.recurrent_hidden_states,
                    'actions': rollouts.actions,
                    'masks': rollouts.masks,
                    'bad_masks': rollouts.bad_masks,
                    'value_preds': rollouts.value_preds,
                }
                sf_buffer = conspecfunction.rollouts.retrieve_SFbuffer()
                conspec_rollouts = {
                    'obs': sf_buffer[0],
                    'rewards': sf_buffer[5],
                    'hidden_states': sf_buffer[1],
                    'actions': sf_buffer[3],
                    'masks': sf_buffer[2],
                    'bad_masks': sf_buffer[2],
                    'value_preds': sf_buffer[4],
                }
                conspec_rollouts_frozen = {}
                for i in range(args.num_prototypes):
                    sf_buffer = conspecfunction.rollouts.retrieve_SFbuffer_frozen(i)
                    conspec_rollouts_frozen[i] = {
                        'obs': sf_buffer[0],
                        'rewards': sf_buffer[5],
                        'hidden_states': sf_buffer[1],
                        'actions': sf_buffer[3],
                        'masks': sf_buffer[2],
                        'bad_masks': sf_buffer[2],
                        'value_preds': sf_buffer[4],
                    }
                tensor_proto_list = [p.data for p in conspecfunction.prototypes.prototypes]
                model_checkpoint = {
                    'epoch': j,
                    'encoder_state_dict': conspecfunction.encoder.state_dict(),
                    'actor_critic_state_dict': actor_critic.state_dict(),
                    'optimizer_conspec_state_dict': conspecfunction.optimizerConSpec.state_dict(),
                    'optimizer_ppo_state_dict': agent.optimizer.state_dict(),
                    'prototypes_state_dict': tensor_proto_list,
                    'prototypes': conspecfunction.prototypes.prototypes.state_dict(),
                    'layers1': conspecfunction.prototypes.layers1.state_dict(),
                    'layers2': conspecfunction.prototypes.layers2.state_dict(),
                    }
                cos_checkpoint = {
                    'cos_max_scores' : conspecfunction.rollouts.cos_max_scores, 
                    'max_indices' : conspecfunction.rollouts.max_indx,
                    'cos_scores' : conspecfunction.rollouts.cos_scores,
                    # 'cos_success' : conspecfunction.rollouts.cos_score_pos,
                    # 'cos_failure' : conspecfunction.rollouts.cos_score_neg,
                }
                        
                print(f'saving checkpoints for epoch {j}...')
                checkpoint_path = os.path.join(full_path, f'model_checkpoint_epoch_{j}.pth')
                buffer_path = os.path.join(full_path, f'buffer_epoch_{j}.pth')
                conspec_rollouts_path = os.path.join(full_path, f'conspec_rollouts_epoch_{j}.pth')
                conspec_rollouts_frozen_path = os.path.join(full_path, f'conspec_rollouts_frozen_epoch_{j}.pth')
                cos_path = os.path.join(full_path, f'cos_sim_epoch_{j}.pth')

                torch.save(model_checkpoint, checkpoint_path)
                print('model checkpoint saved')

                torch.save(buffer, buffer_path)
                print('buffer saved')

                torch.save(conspec_rollouts, conspec_rollouts_path)
                print('success/failure buffers saved')

                torch.save(conspec_rollouts_frozen, conspec_rollouts_frozen_path)
                print('frozen buffers saved')

                torch.save(cos_checkpoint, cos_path)
                print('cosine similarity saved')

                checkpoint_saved = True

    wandb.finish()


if __name__ == "__main__":
    main()
