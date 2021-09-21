import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class Dn(nn.Module):
    def __init__(self, n_actions, n_states, params):
        print("use deep net")
        super(Dn,self).__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = params["lr"]
        self.n_hidden=params["n_hidden"]

        self.fc1 = nn.Linear(self.n_states, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.loss = nn.L1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def state_to_list(self, state):
        state_list = [0] * self.n_states
        state_list[state] = 1
        return state_list

    
    def action_to_list(self, action):
        state_list = [0] * self.n_states
        state_list[action] = 1
        return state_list

    
    def Q(self, state, action):
        state_list = self.state_to_list(state)
        data_tensor = T.Tensor(state_list)
        layer1 = F.relu(self.fc1(data_tensor))
        layer2 = self.fc2(layer1)
        return layer2[action]

    
    def forward(self, state):
        state_list = self.state_to_list(state)
        data_tensor = T.Tensor(state_list)
        layer1 = F.relu(self.fc1(data_tensor))
        layer2 = self.fc2(layer1)
        return layer2


    def update_weight(self, delta, state, action):
        self.optimizer.zero_grad()

        predictions = self.Q(state, action)
        predictions[predictions!=0] = 0

        cost = self.loss(predictions, delta)
        cost.backward()
        self.optimizer.step()
