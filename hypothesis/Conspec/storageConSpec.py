import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, 
                 recurrent_hidden_state_size, num_prototypes, SF_buffer_size):

        '''
        contains the current minibatch, as well as the FROZEN success and failure memory buffers
        for each trajectory, the observations, rewards, recurrent hidden states, actions, and masks are stored.
        e.g. obs_batch_frozen_S = the observations for the frozen success buffer
        '''
        ################### MAIN MEMORY BUFFER  ###################
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)

        self.num_processes = num_processes
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.rewardsORIG = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.obs_shape = obs_shape
        self.step = 0

        self.recurrent_hidden_state_size = recurrent_hidden_state_size
        self.num_prototypes = num_prototypes
        self.success = SF_buffer_size
        self.success_sample = SF_buffer_size

        ################### FROZEN MEMORY BUFFERS  ###################
        self.obs_batch_frozen_S = [None] * self.num_prototypes
        self.r_batch_frozen_S = [None] * self.num_prototypes
        self.recurrent_hidden_statesbatch_frozen_S = [None] * self.num_prototypes
        self.act_batch_frozen_S = [None] * self.num_prototypes
        self.masks_batch_frozen_S = [None] * self.num_prototypes
        self.step_frozen_S = 0

        self.obs_batch_frozen_F = [None] * self.num_prototypes
        self.r_batch_frozen_F = [None] * self.num_prototypes
        self.recurrent_hidden_statesbatch_frozen_F = [None] * self.num_prototypes
        self.act_batch_frozen_F = [None] * self.num_prototypes
        self.masks_batch_frozen_F = [None] * self.num_prototypes
        self.step_frozen_F = 0
        for i in range(self.num_prototypes):
            self.obs_batch_frozen_S[i] = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
            self.r_batch_frozen_S[i] = torch.zeros(self.num_steps, self.success, 1)
            self.recurrent_hidden_statesbatch_frozen_S[i] = torch.zeros(self.num_steps + 1, self.success,
                                                                     self.recurrent_hidden_state_size)
            self.act_batch_frozen_S[i] = torch.zeros(self.num_steps, self.success, action_shape)
            self.act_batch_frozen_S[i] = self.act_batch_frozen_S[i].long()
            self.masks_batch_frozen_S[i] = torch.zeros(self.num_steps + 1, self.success, 1)

            self.obs_batch_frozen_F[i] = torch.zeros(self.num_steps + 1, self.success, *self.obs_shape)
            self.r_batch_frozen_F[i] = torch.zeros(self.num_steps, self.success, 1)
            self.recurrent_hidden_statesbatch_frozen_F[i] = torch.zeros(self.num_steps + 1, self.success,
                                                                     self.recurrent_hidden_state_size)
            self.act_batch_frozen_F[i] = torch.zeros(self.num_steps, self.success, action_shape)
            self.act_batch_frozen_F[i] = self.act_batch_frozen_F[i].long()
            self.masks_batch_frozen_F[i] = torch.zeros(self.num_steps + 1, self.success, 1)

        self.prototypesUsed = torch.zeros(self.num_prototypes,)
        self.count_prototypes_timesteps_criterion = torch.zeros(self.num_prototypes,)
        
    def calc_total_reward(self, contrastval):
        self.rewardsORIG = torch.clone(self.rewards)
        self.rewards = self.rewards  + contrastval.unsqueeze(-1)
        return self.rewards


    def to(self, device):
        '''just adding cuda to all the memory buffers'''
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        
        for i in range(self.num_prototypes):
            self.obs_batch_frozen_S[i] = self.obs_batch_frozen_S[i].to(device)
            self.r_batch_frozen_S[i] = self.r_batch_frozen_S[i].to(device)
            self.recurrent_hidden_statesbatch_frozen_S[i] = self.recurrent_hidden_statesbatch_frozen_S[i].to(device)
            self.act_batch_frozen_S[i] = self.act_batch_frozen_S[i].to(device)
            self.masks_batch_frozen_S[i] = self.masks_batch_frozen_S[i].to(device)

            self.obs_batch_frozen_F[i] = self.obs_batch_frozen_F[i].to(device)
            self.r_batch_frozen_F[i] = self.r_batch_frozen_F[i].to(device)
            self.recurrent_hidden_statesbatch_frozen_F[i] = self.recurrent_hidden_statesbatch_frozen_F[i].to(device)
            self.act_batch_frozen_F[i] = self.act_batch_frozen_F[i].to(device)
            self.masks_batch_frozen_F[i] = self.masks_batch_frozen_F[i].to(device)
            

    def insert(self, obs, recurrent_hidden_states, actions,  rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def insert_trajectory_batch(self, obs, recurrent_hidden_states, actions, rewards, masks):
        self.obs = obs
        self.recurrent_hidden_states = recurrent_hidden_states
        self.actions = actions
        self.rewards=rewards
        self.masks=masks
        self.bad_masks=masks.clone()
        # self.step = (self.step + 1) % self.num_steps

    def retrieve_batch(self):
        obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = self.obs[:-1]
        recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
            -1, self.recurrent_hidden_states.size(-1))
        actions_batch = self.actions.view(-1,
                                          self.actions.size(-1))
        masks_batch = self.masks[:-1].view(-1, 1)
        reward_batch = self.rewards.squeeze()

        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch

    def releasefrozenSF(self, i):
        permS = torch.randperm(self.success)
        permF = torch.randperm(self.success)
        permS = permS[:self.success_sample]
        permF = permF[:self.success_sample]
        rew_batch = torch.cat((self.r_batch_frozen_S[i][:, permS], self.r_batch_frozen_F[i][:, permF]), dim=1)
        obs_batch = torch.cat((self.obs_batch_frozen_S[i][:, permS], self.obs_batch_frozen_F[i][:, permF]), dim=1)
        recurrent_hidden_states = torch.cat((self.recurrent_hidden_statesbatch_frozen_S[i][:, permS],
                                             self.recurrent_hidden_statesbatch_frozen_F[i][:, permF]), dim=1)
        act_batch = torch.cat((self.act_batch_frozen_S[i][:, permS], self.act_batch_frozen_F[i][:, permF]), dim=1)
        masks_batch = torch.cat((self.masks_batch_frozen_S[i][:, permS], self.masks_batch_frozen_F[i][:, permF]), dim=1)
        return obs_batch, rew_batch, recurrent_hidden_states, act_batch, masks_batch

    def retrieve_SFbuffer_frozen(self, i):
        '''retrieving memories from the frozen success/failure buffers'''
        obs_batchx, rew_batchx, recurrent_hidden_statesx, act_batchx, masks_batchx = self.releasefrozenSF(i)
        obs_batch = obs_batchx[:-1].view(-1, *self.obs.size()[2:])
        obs_batchorig = obs_batchx[:-1]
        recurrent_hidden_states_batch = recurrent_hidden_statesx[:-1].view(-1, self.recurrent_hidden_states.size(-1))
        actions_batch = act_batchx.view(-1, self.actions.size(-1))
        masks_batch = masks_batchx[:-1].view(-1, 1)
        reward_batch = rew_batchx.squeeze()  # [:-1]#.view(-1, 1)
        return obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, obs_batchorig, reward_batch

    def load_SFbuffer(self, buffer_path):
        """
        Load the SF buffer from a .pth file.
        """
        buffer_data = torch.load(buffer_path)

        # Reshape and assign the loaded data
        obs_batchx = buffer_data['obs'].view(self.num_steps, -1, *self.obs_shape)
        rew_batchx = buffer_data['rewards'].view(self.num_steps, -1, 1)
        recurrent_hidden_statesx = buffer_data['hidden_states'].view(self.num_steps, -1, self.recurrent_hidden_state_size)
        act_batchx = buffer_data['actions'].view(self.num_steps, -1, 1)
        masks_batchx = buffer_data['masks'].view(self.num_steps, -1, 1)

        # Split the data into success and failure buffers
        split_point = obs_batchx.size(1) // 2
        self.obs_batchS[:-1] = obs_batchx[:, :split_point]
        self.obs_batchF[:-1] = obs_batchx[:, split_point:]
        self.r_batchS = rew_batchx[:, :split_point]
        self.r_batchF = rew_batchx[:, split_point:]
        self.recurrent_hidden_statesS[:-1] = recurrent_hidden_statesx[:, :split_point]
        self.recurrent_hidden_statesF[:-1] = recurrent_hidden_statesx[:, split_point:]
        self.act_batchS = act_batchx[:, :split_point]
        self.act_batchF = act_batchx[:, split_point:]
        self.masks_batchS[:-1] = masks_batchx[:, :split_point]
        self.masks_batchF[:-1] = masks_batchx[:, split_point:]

        self.stepS = split_point
        self.stepF = split_point

    def load_SFbuffer_frozen(self, frozen_buffer_path):
        """
        Load the frozen SF buffer from a .pth file.
        """
        frozen_buffer_data = torch.load(frozen_buffer_path)

        for i in range(self.num_prototypes):
            prototype_data = frozen_buffer_data[i]
            
            obs_batchx = prototype_data['obs'].view(self.num_steps, -1, *self.obs_shape)
            rew_batchx = prototype_data['rewards'].view(self.num_steps, -1, 1)
            recurrent_hidden_statesx = prototype_data['hidden_states'].view(self.num_steps, -1, self.recurrent_hidden_state_size)
            act_batchx = prototype_data['actions'].view(self.num_steps, -1, 1)
            masks_batchx = prototype_data['masks'].view(self.num_steps, -1, 1)

            # Split the data into success and failure buffers
            split_point = obs_batchx.size(1) // 2

            self.obs_batch_frozen_S[i][:-1] = obs_batchx[:, :split_point]
            self.obs_batch_frozen_F[i][:-1] = obs_batchx[:, split_point:]
            self.r_batch_frozen_S[i] = rew_batchx[:, :split_point]
            self.r_batch_frozen_F[i] = rew_batchx[:, split_point:]
            self.recurrent_hidden_statesbatch_frozen_S[i][:-1] = recurrent_hidden_statesx[:, :split_point]
            self.recurrent_hidden_statesbatch_frozen_F[i][:-1] = recurrent_hidden_statesx[:, split_point:]
            self.act_batch_frozen_S[i] = act_batchx[:, :split_point]
            self.act_batch_frozen_F[i] = act_batchx[:, split_point:]
            self.masks_batch_frozen_S[i][:-1] = masks_batchx[:, :split_point]
            self.masks_batch_frozen_F[i][:-1] = masks_batchx[:, split_point:]

        self.step_frozen_S = split_point
        self.step_frozen_F = split_point