import numpy as np
import math
from dwave_qbsolv import QBSolv
from neal.sampler import SimulatedAnnealingSampler

class Dbm():
    def __init__(self, n_actions, n_states, n_hidden, lr, beta, init_sd):
        print("use dbm")
        self.lr = lr
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.beta = beta
        self.W1 = np.random.normal(0, init_sd, (self.n_states, self.n_hidden))
        self.W2 = np.random.normal(0, init_sd, (self.n_actions, self.n_hidden))


    def get_h(self, s , a):

        h = {}

        for i in range(self.n_hidden):
            h[(i, i)] = self.W1[s][i] + self.W2[a][i]

        J = {}
        
        response = SimulatedAnnealingSampler().sample_ising(h, J, num_reads=500, num_sweeps=200, beta_range=[0.1, 30])
        #response = SimulatedAnnealingSampler().sample_ising(h, J, num_reads=20, num_sweeps=200, beta_range=[0.1, 15])
        samples = list(response.samples())
        h = []
        for i in range(self.n_hidden):
            solutions = []
            for sample in samples:
                solutions.append( (sample[(i,i)] + 1)/2)
            h_i = np.mean(solutions)
            if h_i >= 1:
                h_i = 0.999999
            if h_i <= 0:
                h_i = 0.000001
            h.append(h_i)

        #print(h)

        return h


    def Q(self, s, a):
        h = self.get_h(s, a)
        sum1 = np.dot(self.W1[s], h)
        sum2 = np.dot(self.W2[a], h)
        sum3 = 0
        for item in h:
            sum3 += item * math.log(item) + (1 - item) * math.log(1 - item)
        result = sum1 + sum2 - 1/self.beta * sum3
        return result


    def update_weight(self, delta, state, action):
        h = self.get_h(state, action)
        with np.nditer(self.W1[state], op_flags=['readwrite']) as W_s:
            index = 0
            for w_sh in W_s:
                w_sh[...] += delta * h[index]
                index += 1
        with np.nditer(self.W2[action], op_flags=['readwrite']) as W_a:
            index = 0
            for w_ah in W_a:
                w_ah[...] += delta * h[index]
                index += 1