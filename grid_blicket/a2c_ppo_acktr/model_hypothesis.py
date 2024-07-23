import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

class Hypothesis_Policy(nn.Module):
    def __init__(self, obs_shape, action_space, num_prototypes, base=None, base_kwargs=None):
        super(Hypothesis_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = ImpalaModel
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], num_prototypes, **base_kwargs)

        self.dist = Categorical(self.base.output_size, action_space.n)


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, hypothesis, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, hypothesis, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, hypothesis, rnn_hxs, masks):
        value, _, _ = self.base(inputs, hypothesis, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, hypothesis, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, hypothesis, rnn_hxs, masks)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
    
    def retrieve_hiddens(self, inputs, hypothesis, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, hypothesis, rnn_hxs, masks)
        return actor_features


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        #print(wefw)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    def __init__(self, in_channels, num_prototypes, observation_space, recurrent=False, hidden_size=256):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=8)
        self.block2 = ImpalaBlock(in_channels=8, out_channels=16)
        self.block3 = ImpalaBlock(in_channels=16, out_channels=16)
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self._get_conv_output(sample_input).shape[1]

        self.fc = nn.Linear(in_features=n_flatten + num_prototypes, out_features=hidden_size)

        self.output_dim = hidden_size
        self.apply(xavier_uniform_init)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

        self._recurrent = recurrent
        self._hidden_size = hidden_size

    def _get_conv_output(self, x):
        x = self.block1(x / 255.0)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        return Flatten()(x)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x, hypothesis, rnn_hxs, masks):
        x = self.block1(x / 255.0)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = torch.cat([x, hypothesis], dim=1)  # Concatenate CNN output with hypothesis
        x = self.fc(x)
        x = nn.ReLU()(x)

        return self.critic_linear(x), x, rnn_hxs
    


