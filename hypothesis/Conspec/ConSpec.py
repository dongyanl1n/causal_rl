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
            """
        num_actions = actionspace.n
        self.encoder = EncoderConSpec(
            obsspace.shape,
            num_actions,
            base_kwargs={'recurrent': args.recurrent_policy,  # only to ensure it has the same hidden size as RL model; forward pass through CNN actually does not include GRU
                     'observation_space': obsspace})  # envs.observation_space.shape,
        self.encoder.to(device)
        self.intrinsicR_scale = args.intrinsicR_scale
        self.num_procs = args.num_processes
        self.num_prototypes = args.num_prototypes
        self.seed = args.seed
        self.rollouts = RolloutStorage(args.num_steps, self.num_procs, obsspace.shape,
                              self.encoder.recurrent_hidden_state_size, self.num_prototypes, args.SF_buffer_size)
        self.prototypes = prototypes(input_size=self.encoder.recurrent_hidden_state_size, hidden_size=1010, 
                                     num_prototypes=self.num_prototypes, device=device)
        self.device = device
        self.prototypes.to(device)
        self.cos_score_threshold = args.cos_score_threshold
        self.roundhalf = args.roundhalf

        self.listparams = list(self.encoder.parameters()) + list(self.prototypes.parameters())

    def store_memories(self,image_to_store, memory_to_store, action_to_store, reward_to_store, masks_to_store):
        '''stores the current minibatch of trajectories from the RL agent into the memory buffer for the current minibatch'''
        with torch.no_grad():
            self.rollouts.insert_trajectory_batch(image_to_store, memory_to_store, action_to_store, reward_to_store, masks_to_store)  # store the current minibatch of trajectories in the main buffer
            self.rollouts.to(self.device)

    def calc_cos_scores(self,obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig):
        '''computes the cosine similarity scores'''
        hidden = self.encoder.retrieve_hiddens(obs_batch, 
                                               recurrent_hidden_states_batch, 
                                               masks_batch)  # shape: torch.Size([ep_length*num_rollouts, 512])
        hidden = hidden.view(*obs_batchorig.size()[:2], -1)  # shape: torch.Size([ep_length, num_rollouts, 512])
        return self.prototypes(hidden)

    def calc_intrinsic_reward_v0(self, prototypes_used):
        '''computes the intrinsic reward for the current minibatch of trajectories'''
        with torch.no_grad():
            obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch = self.rollouts.retrieve_batch()
            cos_scores = self.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig)
            cos_scores = cos_scores.view(*obs_batchorig.size()[:2], -1)
            cos_scores = ((cos_scores > self.cos_score_threshold)) * cos_scores  # threshold applied to ignore small score fluctuations
            prototypes_used = torch.tile(torch.reshape(prototypes_used, (1, 1, -1)), (*cos_scores.shape[:2], 1)) # 

            prototypes_used = prototypes_used * self.intrinsicR_scale

            intrinsic_reward = (cos_scores * prototypes_used)
            roundhalf = self.roundhalf  # window size

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
            # intrinsic_reward[intrinsic_reward < 0.] = 0.
            zero_sum = 0.
            for i in range(roundhalf):
                temp = torch.roll(intrinsic_reward, i + 1, dims=0) * (.5 ** (i))
                temp[:i] = 0.
                zero_sum += temp
            intrinsic_reward -= zero_sum
            # intrinsic_reward[intrinsic_reward < 0.] = 0.
            intrinsic_reward = intrinsic_reward.sum(2)
            '''compute the total reward = intrinsic reward + environment reward'''
            return self.rollouts.calc_total_reward(intrinsic_reward), intrinsic_reward


    def calc_intrinsic_reward_eq3(self, hypothesis_batch):
        '''computes the intrinsic reward using equation 3 of the paper'''
        with torch.no_grad():
            obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, _ = self.rollouts.retrieve_batch()
            _, _, _, cos_scores, _ = self.calc_cos_scores(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, -1)
            cos_scores = cos_scores.view(*obs_batchorig.size()[:2], -1)

            # Apply the discount factor gamma and aalculate the difference between current and previous cosine scores
            discounted_diff = self.gamma * cos_scores - torch.roll(cos_scores, shifts=1, dims=0)
            discounted_diff[0] = cos_scores[0]  # Set the first row to the initial scores
            
            discounted_diff = discounted_diff * hypothesis_batch.unsqueeze(0) # apply hypothesis mask: num_processes, num_prototypes

            # Sum over all prototypes (H in the equation)
            summed_diff = discounted_diff.sum(dim=-1)
            
            # Apply the proportionality constant lambda (intrinsicR_scale)
            intrinsic_reward = self.intrinsicR_scale * summed_diff
            
            # Ensure non-negative rewards
            intrinsic_reward = torch.clamp(intrinsic_reward, min=0.0)
            
            '''compute the total reward = intrinsic reward + environment reward'''
            return self.rollouts.calc_total_reward(intrinsic_reward), intrinsic_reward  # adds the intrinsic reward to the environment reward


    def freeze_parameters(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.prototypes.parameters():
            param.requires_grad = False
        for param in self.prototypes.layers1.parameters():
            param.requires_grad = False
        for param in self.prototypes.layers2.parameters():
            param.requires_grad = False