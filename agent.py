import numpy as np
import random
from util import std_dev_from_h
from q_table import Q_table
from dn import Dn
from rbm import Rbm
from dbm import Dbm
from dbm_2 import Dbm_2
from qbm import Qbm

class Agent():
    def __init__(self, env, params):
        self.env = env
        self.gamma = params["gamma"]
        self.n_actions = self.env.n_actions
        self.n_states = self.env.rows * self.env.cols
        self.params = params
        if params["method"] == 'q_table':
            self.method = Q_table(self.n_actions, self.n_states, params)
        if params["method"] == 'dn':
            self.method = Dn(self.n_actions, self.n_states, params)
        if params["method"] == 'rbm':
            self.method = Rbm(self.n_actions, self.n_states, params)
        if params["method"] == 'dbm':
            self.method = Dbm(self.n_actions, self.n_states, params)
            std_dev_from_h(self)
        if params["method"] == 'dbm_2':
            print("under cnstruction")
            self.method = Dbm_2(self.n_actions, self.n_states, params)
            std_dev_from_h(self)
        if params["method"] == 'qbm':
            self.method = Qbm(self.n_actions, self.n_states, params)
            std_dev_from_h(self)


    def choose_action(self, state):
        actions = np.zeros(4)
        for action in range(self.n_actions):
            actions[action] = self.method.Q(state, action)
        action = np.argmax(actions)
        return action


    def print_action_choices(self, state):
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
            Qs2a2 = 0
        elif self.env.map[self.env.state[0]][self.env.state[1]] == -1:
            Qs2a2 = 0
        else:
            print("learn error occured")
        delta = reward + self.gamma * Qs2a2 - Qs1a1
        self.method.update_weight(delta, state, action)


    def print_policy(self):
        for i in range(self.env.map.shape[0]): # iterate over rows
            for j in range(self.env.map[i].shape[0]): # iterate over cols
                if self.env.map[i][j] == -1:
                    print("o", end=' ')
                elif self.env.map[i][j] == 1:
                    print("x", end=' ')
                elif self.env.map[i][j] == 0:
                    state = j + self.env.cols * i
                    action = self.choose_action(state)
                    symbol = self.env.action_symbol(action)
                    print(symbol, end=' ')
                else:
                    print("error in print policy")
            print()