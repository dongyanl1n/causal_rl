import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

np.set_printoptions(threshold=10_000)
torch.set_printoptions(threshold=10_000)

class prototypes(nn.Module):
    def __init__(self,
                 input_size, hidden_size, num_prototypes, device):
        super(prototypes, self).__init__()
        self.num_prototypes = num_prototypes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.layers1 = nn.ModuleList([nn.Sequential(
            (nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)), nn.ReLU(),
        ) for i in range(num_prototypes)])
        self.layers2 = nn.ModuleList([nn.Sequential(
            (nn.Linear(in_features=hidden_size, out_features=hidden_size,bias=False)), nn.ReLU(),
        ) for i in range(num_prototypes)])

        self.prototypes = nn.ParameterList([nn.Parameter(torch.randn(1, self.hidden_size) * 1.) for i in range(num_prototypes)])
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, hidden):
        # hidden.shape = torch.Size([ep_length, num_rollouts, 512])
        out1 = [None] * self.num_prototypes
        cos_scores = [None] * self.num_prototypes
        
        for i in range(self.num_prototypes):
            out1[i] = self.layers2[i](self.layers1[i](hidden))  # pass z_t through g_theta
            s1, s2, s3 = out1[i].shape  # s1 = ep_length, s2 = num_processes, s3 = 1010
            prototypes = self.prototypes[i].reshape(1, 1, -1).repeat(s1, 1, 1)  # repeat prototype vector for ep_length times --> ep_length, 1, 1010
            cos_scores[i] = self.cos(out1[i], prototypes) # compare current embedding with prototype at each time --> ep_length, num_processes
        cos_scores = torch.stack(cos_scores, dim=2)  # ep_length, num_processes, num_prototypes
        # cos_max, indices = torch.max(cos_scores, dim=0)  # t that maximizes the cosine similarity
        return cos_scores


    def get_prototype_data(self):
        prototypes = {}
        for i, param in enumerate(self.prototypes):
        # Convert parameter to numpy array and flatten it
            param_data = param.detach().cpu().numpy().flatten() # each prototype shape is (1010,)
            prototypes['proto '+str(i)] = param_data
        return prototypes 