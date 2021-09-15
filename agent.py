import numpy as np
import random
from rbm import Rbm
from dbm import Dbm
from dbm_2 import Dbm_2

class Agent():
    def __init__(self, env, method, n_hidden, lr, gamma, beta, init_sd):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.n_actions = self.env.n_actions
        self.n_states = self.env.rows * self.env.cols
        self.n_hidden = n_hidden
        if method == 'rbm':
            self.method = Rbm(self.n_actions, self.n_states, n_hidden, lr, beta, init_sd)
        if method == 'dbm':
            self.method = Dbm(self.n_actions, self.n_states, n_hidden, lr, beta, init_sd)
        if method == 'dbm_2':
            self.method = Dbm_2(self.n_actions, self.n_states, n_hidden, lr, beta, init_sd)


    def choose_action(self, state):
        actions = np.zeros(4)
        for a in range(self.n_actions):
            actions[a] = self.method.Q(state,a)
        action = np.argmax(actions)
        return action


    def print_action_choices(self, state):#
        actions = np.zeros(4)
        for a in range(self.n_actions):
            actions[a] = self.method.Q(state,a)
        print(actions)
        action = np.argmax(actions)
        print(action)


    def get_epsilon(self, runs, run):
        return max(1 - run / runs, 0.01)


    def choose_eps_greedy_action(self, state, runs, run):
        epsilon = self.get_epsilon(runs, run)
        if np.random.random() > epsilon:
            action = self.choose_action(state)
        else:
            action = random.randrange(0, self.n_actions)
        return action, epsilon


    def learn(self, state, action, reward, state_):
        Qs1a1 = self.method.Q(state,action)
        if self.env.map[self.env.state[0]][self.env.state[1]] == 0:
            action_ = self.choose_action(state_)
            Qs2a2 = self.method.Q(state_,action_)
        elif self.env.map[self.env.state[0]][self.env.state[1]] == 1:
            Qs2a2 = 1
        elif self.env.map[self.env.state[0]][self.env.state[1]] == -1:
            Qs2a2 = 0
        else:
            print("learn error occured")
        delta = self.lr * (reward + self.gamma * Qs2a2 - Qs1a1)
        self.method.update_weight(delta, state, action)


    def print_policy(self):
        for i in range(self.env.map.shape[0]): # iterate over rows
            for j in range(self.env.map[i].shape[0]): # iterate over cols
                state = j + self.env.cols * i
                action = self.choose_action(state)
                symbol = self.env.action_symbol(action)
                print(symbol, end=' ')
            print()