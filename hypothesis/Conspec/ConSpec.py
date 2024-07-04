import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
from .storageConSpec import RolloutStorage
from .prototype import prototypes
from .modelConSpec import EncoderConSpec
import matplotlib.pyplot as plt



def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

class ConSpec(nn.Module):
    def __init__(self, args, obsspace,  actionspace, device):
        super(ConSpec, self).__init__()

        """Class that contains everything needed to implement ConSpec

            Args:
                args: the arguments
                obsspace: observation space. use obsspace.shape to get shape of the observation space
                actionspace: action space. use actionspace.n to get number of actions
                device: the device

            Usage:
                self.encoder = the encoder used to take in observations from the enviornment and output latents
                intrinsicR_scale = lambda, from the paper
                self.num_processes = minibatch size
                self.num_prototypes = number of prototypes
                self.rollouts = contains all the memory buffers, including success and failure memory buffers
                self.prototypes = contains the prototype vectors as well as their projection MLPs g_theta from the paper
                self.optimizerConSpec = the optimizer for the encoder + the prototypes
            """
        num_actions = actionspace.n
        self.encoder = EncoderConSpec(
            obsspace.shape,
            num_actions,
            base_kwargs={'recurrent': args.recurrent_policy,  # only to ensure it has the same hidden size as RL model; forward pass through CNN actually does not include GRU
                     'observation_space': obsspace})  # envs.observation_space.shape,
        self.encoder.to(device)
        self.intrinsicR_scale = args.intrinsicR_scale
        self.gamma = args.gamma
        self.num_procs = args.num_processes
        self.num_prototypes = args.num_prototypes
        self.seed = args.seed
        self.rollouts = RolloutStorage(args.num_steps, self.num_procs, obsspace.shape,
                              self.encoder.recurrent_hidden_state_size, self.num_prototypes)
        self.prototypes = prototypes(input_size=self.encoder.recurrent_hidden_state_size, hidden_size=1010, 
                                     num_prototypes=self.num_prototypes, device=device)
        self.device = device
        self.prototypes.to(device)

        self.listparams = list(self.encoder.parameters()) + list(self.prototypes.parameters())
        self.optimizerConSpec = optim.Adam(self.listparams, lr=args.lrConSpec, eps=args.eps)

    def store_memories(self,image_to_store, memory_to_store, action_to_store, reward_to_store, masks_to_store):
        '''stores the current minibatch of trajectories from the RL agent into the memory buffer for the current minibatch, as well as the success (pos) and failure (neg) memory buffers'''
        with torch.no_grad():
            self.rollouts.insert_trajectory_batch(image_to_store, memory_to_store, action_to_store, reward_to_store, masks_to_store)  # store the current minibatch of trajectories in the main buffer
            self.rollouts.to(self.device)
            self.rollouts.addPosNeg('pos', self.device) ###add to the positive memory buffer
            self.rollouts.addPosNeg('neg', self.device) ###add to the negative memory buffer

    def calc_cos_scores(self,obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, prototype_number):
        '''computes the cosine similarity scores'''
        # shapes: 
        # obs_batch = torch.Size([ep_length*num_processes, 3, 56, 56])
        # recurrent_hidden_states_batch = torch.Size([ep_length*num_processes, 1])
        # masks_batch = torch.Size([ep_length*num_processes, 1])
        # actions_batch = torch.Size([ep_length*num_processes, 1])
        # obs_batchorig = torch.Size([ep_length, num_processes, 3, 56, 56])
        hidden = self.encoder.retrieve_hiddens(obs_batch, 
                                               recurrent_hidden_states_batch, 
                                               masks_batch)  # shape: torch.Size([ep_length*num_processes, 512])
        hidden = hidden.view(*obs_batchorig.size()[:2], -1)  # shape: torch.Size([ep_length, num_processes, 512])
        return self.prototypes(hidden, prototype_number)

    def calc_intrinsic_reward(self, hypothesis_batch):
        '''computes the intrinsic reward for the current minibatch of trajectories'''
        prototypes_used, count_prototypes_timesteps_criterion = self.rollouts.retrieve_prototypes_used()  # [0,0,0,0,0,0,0,0]
        prototypes_used = prototypes_used.to(device=self.device)
        with torch.no_grad():
            obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = self.rollouts.retrieve_batch()
            _, _, _, cos_scores, _ = self.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, -1)
            cos_scores = cos_scores.view(*obs_batchorig.size()[:2], -1)
            cos_scores = ((cos_scores > 0.6)) * cos_scores  # threshold of 0.6 applied to ignore small score fluctuations
            prototypes_used = torch.tile(torch.reshape(prototypes_used, (1, 1, -1)), (*cos_scores.shape[:2], 1)) # 

            prototypes_used = prototypes_used * self.intrinsicR_scale

            intrinsic_reward = (cos_scores * prototypes_used)
            roundhalf = 3  # window size = 7

            '''find the max rewards in each rolling average (equation 2 of the manuscript)'''
            rolling_max = []
            for i in range(roundhalf * 2 + 1):
                temp = torch.roll(intrinsic_reward, i - roundhalf, dims=0)
                if i - roundhalf > 0:
                    temp[:(i - roundhalf)] = 0.
                rolling_max.append(temp)
            rolling_max = torch.stack(rolling_max, dim=0)
            rolling_max, _ = torch.max(rolling_max, dim=0)
            allvaluesdifference = intrinsic_reward - rolling_max
            intrinsic_reward[allvaluesdifference < 0.] = 0.
            intrinsic_reward[intrinsic_reward < 0.] = 0.
            zero_sum = 0.
            for i in range(roundhalf):
                temp = torch.roll(intrinsic_reward, i + 1, dims=0) * (.5 ** (i))
                temp[:i] = 0.
                zero_sum += temp
            intrinsic_reward -= zero_sum
            intrinsic_reward[intrinsic_reward < 0.] = 0.
            intrinsic_reward = intrinsic_reward * hypothesis_batch.unsqueeze(0)  # apply hypothesis mask: num_processes, num_prototypes
            intrinsic_reward = intrinsic_reward.sum(2)
            '''compute the total reward = intrinsic reward + environment reward'''
            return self.rollouts.calc_total_reward(intrinsic_reward), intrinsic_reward  # adds the intrinsic reward to the environment reward

    def calc_intrinsic_reward_eq3(self, hypothesis_batch):
        '''computes the intrinsic reward using equation 3 of the paper'''
        prototypes_used = self.rollouts.retrieve_prototypes_used()[0].to(device=self.device)
        with torch.no_grad():
            obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, _ = self.rollouts.retrieve_batch()
            _, _, _, cos_scores, _ = self.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, -1)
            cos_scores = cos_scores.view(*obs_batchorig.size()[:2], -1)
            prototypes_used = torch.tile(torch.reshape(prototypes_used, (1, 1, -1)), (*cos_scores.shape[:2], 1)) # repeat prototype vector for ep_length times

            # Apply the discount factor gamma and aalculate the difference between current and previous cosine scores
            discounted_diff = self.gamma * cos_scores - torch.roll(cos_scores, shifts=1, dims=0)
            discounted_diff[0] = cos_scores[0]  # Set the first row to the initial scores

            discounted_diff = discounted_diff * prototypes_used  # Apply the prototypes used mask  # shape: torch.Size([ep_length, num_processes, num_prototypes])
            
            discounted_diff = discounted_diff * hypothesis_batch.unsqueeze(0) # apply hypothesis mask: num_processes, num_prototypes

            # Sum over all prototypes (H in the equation)
            summed_diff = discounted_diff.sum(dim=-1)
            
            # Apply the proportionality constant lambda (intrinsicR_scale)
            intrinsic_reward = self.intrinsicR_scale * summed_diff
            
            # Ensure non-negative rewards
            intrinsic_reward = torch.clamp(intrinsic_reward, min=0.0)
            
            '''compute the total reward = intrinsic reward + environment reward'''
            return self.rollouts.calc_total_reward(intrinsic_reward), intrinsic_reward  # adds the intrinsic reward to the environment reward
    
    def update_conspec(self):
        '''trains the ConSpec module'''
        prototypes_used, count_prototypes_timesteps_criterion = self.rollouts.retrieve_prototypes_used()
        cos_scores_pos = []
        cos_scores_neg = []
        costCL = 0
        if self.rollouts.stepS > self.rollouts.success - 1:
            ########################
            for j in range(self.num_prototypes):
                if prototypes_used[j] > 0.5:
                    # print('frozen prototypes used', j, prototypes_used[j])
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = self.rollouts.retrieve_SFbuffer_frozen(
                        j)
                else:
                    # print('still using SF buffer not the frozen one')
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = self.rollouts.retrieve_SFbuffer()

                cos_max_score, max_inds, cost_prototype, cos_scores, cos_score_sf = self.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch,
                                                        actions_batch, obs_batchorig, j)    

                costCL += cost_prototype
                cos_scores_pos.append(cos_score_sf[0][j].detach().cpu())
                cos_scores_neg.append(cos_score_sf[1][j].detach().cpu())
                self.rollouts.cos_max_scores = cos_max_score
                self.rollouts.max_indx = max_inds
                self.rollouts.cos_scores = cos_scores
                # self.rollouts.cos_score_pos = cos_score_sf[0]
                # self.rollouts.cos_score_neg = cos_score_sf[1]

            for i in range(self.num_prototypes):
                if (cos_scores_pos[i] - cos_scores_neg[i] > 0.6) and cos_scores_pos[i] > 0.6:
                    count_prototypes_timesteps_criterion[i] += 1  # stop updating the prototype if the criterion is met
                else:
                    count_prototypes_timesteps_criterion[i] = 0 # reset the counter if the criterion is not met
                if count_prototypes_timesteps_criterion[i] > 25 and prototypes_used[i] < 0.1:
                    prototypes_used[i] = 1.
                    self.rollouts.store_frozen_SF(i)

            self.optimizerConSpec.zero_grad()
            costCL.backward()  # learn encoder and prototypes
            self.optimizerConSpec.step()
        self.rollouts.store_prototypes_used(prototypes_used, count_prototypes_timesteps_criterion)
        torch.cuda.empty_cache()

    def do_everything(self, obstotal,  recurrent_hidden_statestotal, actiontotal,rewardtotal, maskstotal, hypothesis_batch):
        '''function for doing all the required conspec functions above, in order'''
        self.store_memories(obstotal, recurrent_hidden_statestotal, actiontotal, rewardtotal, maskstotal)  # store in main buffer and pos/neg buffer  # ep_length, num_processes, *
        # rewardtotal_intrisic_extrinsic, reward_intrinsic = self.calc_intrinsic_reward(hypothesis_batch)
        rewardtotal_intrisic_extrinsic, reward_intrinsic = self.calc_intrinsic_reward_eq3(hypothesis_batch)
        self.update_conspec()
        return rewardtotal_intrisic_extrinsic, reward_intrinsic
    

