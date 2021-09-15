import numpy as np
import math

class Rbm():
    def __init__(self, n_actions, n_states, n_hidden, lr, beta, init_sd):
        print("use rbm")
        self.lr = lr
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_visible = n_actions + n_states
        self.n_hidden = n_hidden
        self.beta = beta
        self.W = np.random.normal(0, init_sd, (self.n_visible, self.n_hidden))


    def sigmoid(self, x):
        if x > 36:
            return 0.999999
        elif x < -36:
            return 0.000001
        else:
            return 1.0 / (1 + math.exp(-x))

    def get_h(self, s , a):
        h = self.W[s] + self.W[a + self.n_states]
        with np.nditer(h, op_flags=['readwrite']) as h_it:
            for h_i in h_it:
                h_i[...] = self.sigmoid(h_i)
        return h

    def Q(self, s, a):
        h = self.get_h(s, a)
        sum1 = np.dot(self.W[s], h)
        sum2 = np.dot(self.W[self.n_states + a], h)
        sum3 = 0
        for item in np.nditer(h):
            sum3 += item * math.log(item) + (1 - item) * math.log(1 - item)
        result = sum1 + sum2 - 1/self.beta * sum3
        return result

    def update_weight(self, delta, state, action):
        h = self.get_h(state, action)
        with np.nditer(self.W[state], op_flags=['readwrite']) as W_s:
            index = 0
            for w_sh in W_s:
                w_sh[...] += delta * h[index]
                index += 1
        with np.nditer(self.W[self.n_states + action], op_flags=['readwrite']) as W_a:
            index = 0
            for w_ah in W_a:
                w_ah[...] += delta * h[index]
                index += 1