import torch
import torch.nn as nn
import torch.optim as optim

use_cuda = True
if torch.cuda.is_available() and use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ActorCriticRNN(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, device):
        super(ActorCriticRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        
        # RNN Layer
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True).to(self.device)
        
        # Actor: Policy network
        self.actor = nn.Linear(hidden_size, action_size).to(self.device)
        self.softmax = nn.Softmax(dim=-1)
        
        # Critic: Value network
        self.critic = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x, hidden):
        x = x.to(self.device)
        hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
        x, new_hidden = self.rnn(x, hidden)
        x = x[-1, :]  # get last RNN output  # note for batched output, this would be x[:, -1, :]
        policy = self.softmax(self.actor(x))
        value = self.critic(x)
        return policy, value, new_hidden

    def init_hidden(self):
        return (torch.zeros(1, self.hidden_size, device=self.device),
                torch.zeros(1, self.hidden_size, device=self.device))  # would be (1, batch_size, self.hidden_size) for batched input



class Core:
    def __init__(self, state_size, action_size, hidden_size, device=device):
        self.learner = ActorCriticRNN(state_size, action_size, hidden_size, device=device)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=0.01)
        self.hidden = self.learner.init_hidden()

    def select_action(self, state):
        state = torch.tensor(state.reshape(1,-1), dtype=torch.float)
        policy, value, new_hidden = self.learner(state, self.hidden)
        self.hidden = new_hidden  # Update the hidden state
        action = policy.multinomial(num_samples=1)
        log_prob = torch.log(policy.squeeze(0)[action])
        return action.item(), value, log_prob

    def learn(self, log_probs, values, rewards, dones):
        self.optimizer.zero_grad()
        rewards = torch.tensor(rewards).to(device)
        returns = self.compute_returns(rewards, dones)
        policy_loss = self.compute_policy_loss(log_probs, returns)
        value_loss = self.compute_value_loss(values, returns)
        loss = policy_loss + value_loss
        loss.backward()
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
        return torch.tensor(returns, device=rewards.device, dtype=torch.float)
    
    def compute_policy_loss(self, log_probs, returns):
        # breakpoint()
        log_probs = torch.stack(log_probs).squeeze()
        return -torch.sum(log_probs * returns)
    
    def compute_value_loss(self, values, returns):
        values = torch.stack(values).squeeze()
        return torch.sum((returns - values) ** 2)
    

