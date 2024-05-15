import torch
import torch.nn as nn
import torch.optim as optim

use_cuda = True
if torch.cuda.is_available() and use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class BaselineActorCriticRNN(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, device):
        super(BaselineActorCriticRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True).to(self.device)
        
        # Single actor head
        self.actor = nn.Linear(hidden_size, action_size).to(self.device)
        self.softmax = nn.Softmax(dim=-1)
        
        # Critic layer
        self.critic = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x, hidden):
        x = x.to(self.device)
        hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
        x, new_hidden = self.rnn(x, hidden)
        x = x[-1, :]  # For batch processing, this would need adjustment if dealing with non-batched inputs
        
        actor_output = self.actor(x)
        policy = self.softmax(actor_output)
        value = self.critic(x)
        
        return policy, value, new_hidden

    def init_hidden(self):
        return (torch.zeros(1, self.hidden_size, device=self.device),
                torch.zeros(1, self.hidden_size, device=self.device))


class BaselineCore:
    def __init__(self, state_size, action_size, hidden_size, lr, device=device):
        self.device = device
        self.learner = BaselineActorCriticRNN(state_size, action_size, hidden_size, device=device)
        self.lr = lr
        # Single optimizer for both actor and critic
        self.optimizer = optim.Adam(
            list(self.learner.actor.parameters()) + list(self.learner.critic.parameters()),
            lr=lr
        )
        self.hidden = self.learner.init_hidden()

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(self.device)
        policy, value, new_hidden = self.learner(state, self.hidden)
        self.hidden = new_hidden  # Update the hidden state
        action = policy.multinomial(num_samples=1).item()
        log_prob = torch.log(policy.squeeze(0)[action])
        return action, value, log_prob

    def learn(self, log_probs, values, rewards, dones, retain_graph=False):
        self.optimizer.zero_grad()
        rewards = torch.tensor(rewards, device=self.device)
        returns = self.compute_returns(rewards, dones)
        policy_loss = self.compute_policy_loss(log_probs, returns)
        value_loss = self.compute_value_loss(values, returns)
        loss = policy_loss + value_loss
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        return loss.item()

    def reset_hidden(self):
        self.hidden = self.learner.init_hidden()

    def compute_returns(self, rewards, dones, gamma=0.99):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float, device=self.device)
    
    def compute_policy_loss(self, log_probs, returns):
        log_probs = torch.stack(log_probs).squeeze()
        return -torch.sum(log_probs * returns)
    
    def compute_value_loss(self, values, returns):
        values = torch.stack(values).squeeze()
        return torch.sum((returns - values) ** 2)

# Example usage:
# state_size = 10  # Example state size
# action_size = 4  # Example action size
# hidden_size = 128  # Example hidden size
# agent = BaselineCore(state_size, action_size, hidden_size)
