import numpy as np
import math
from dwave_qbsolv import QBSolv
from neal.sampler import SimulatedAnnealingSampler

class Dbm_2():
    def __init__(self, n_actions, n_states, n_hidden, lr, beta, init_sd):
        print("use dbm 2")
        self.lr = lr
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.beta = beta
        self.W1 = np.random.normal(0, init_sd, (self.n_states, self.n_hidden))
        self.W2 = np.random.normal(0, init_sd, (self.n_hidden, self.n_hidden))
        self.W3 = np.random.normal(0, init_sd, (self.n_actions, self.n_hidden))


    def get_h(self, s , a):

        Q = {}

        for i in range(self.n_hidden * 2):
            Q[(i, i)] = 0
            for j in range(i):
                    Q[(j, i)] = 0

        for i in range(self.n_hidden):
            Q[(i, i)] = self.W1[s][i]

        for i in range(self.n_hidden):
            for j in range(i):
                    Q[(j, i + self.n_hidden)] = self.W2[i][j]

        for i in range(self.n_hidden):
            Q[(i + self.n_hidden, i + self.n_hidden)] = self.W3[a][i]

        for i in range(self.n_hidden*2):
            if Q[(i, i)] == 0:
                del Q[(i, i)]
            for j in range(i):
                if Q[(j, i)] == 0:
                    del Q[(j, i)]

        # print(Q)
        
        response = SimulatedAnnealingSampler().sample_qubo(Q, num_reads=100)
        samples = list(response.samples())
        h = []
        for i in range(self.n_hidden*2):
            solutions = []
            for sample in samples:
                solutions.append(sample[i])
            h_i = np.mean(solutions)
            if h_i >= 1:
                h_i = 0.999999
            if h_i <= 0:
                h_i = 0.000001
            h.append(h_i)

        # print(h)

        return h


    def Q(self, s, a):
        h = self.get_h(s, a)
        sum1 = np.dot(self.W1[s], h[0:10])
        sum2 = np.dot(self.W3[a], h[10:20])
        sum3 = 0
        for i in range(self.n_hidden):
            sum3 += h[i] * np.dot(self.W2[i], h[10:20])
        sum4 = 0
        for item in h:
            sum4 += item * math.log(item) + (1 - item) * math.log(1 - item)
        result = sum1 + sum2 + sum3 - 1/self.beta * sum4
        return result


    def update_weight(self, delta, state, action):
        delta *= self.lr
        h = self.get_h(state, action)
        with np.nditer(self.W1[state], op_flags=['readwrite']) as W_s:
            index = 0
            for w_sh in W_s:
                w_sh[...] += delta * h[index]
                index += 1
        for i in range(self.n_hidden):
            with np.nditer(self.W2[i], op_flags=['readwrite']) as W_h:
                index = 0
                for w_hh in W_h:
                    w_hh[...] += delta * h[index+self.n_hidden]
                    index += 1
        with np.nditer(self.W3[action], op_flags=['readwrite']) as W_a:
            index = 0
            for w_ah in W_a:
                w_ah[...] += delta * h[index+self.n_hidden]
                index += 1