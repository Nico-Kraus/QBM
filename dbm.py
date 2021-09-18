import numpy as np
import math
from neal.sampler import SimulatedAnnealingSampler

class Dbm():
    def __init__(self, n_actions, n_states, params):
        print("use dbm")
        self.lr = params["lr"]
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_hidden = params["n_hidden"]
        self.beta = params["beta"]
        self.num_reads = params["annealing"]["num_reads"]
        self.num_sweeps = params["annealing"]["num_sweeps"]
        self.beta_range = params["annealing"]["beta_range"]
        self.W1 = np.random.normal(0, params["init_sd"], (self.n_states, self.n_hidden))
        self.W2 = np.random.normal(0, params["init_sd"], (self.n_actions, self.n_hidden))
        self.Wh = np.random.normal(0, params["init_sd"], (self.n_hidden, self.n_hidden))


    def get_h(self, s , a):

        h = {}
        J = {}

        for i in range(self.n_hidden ):
            h[(i, i)] = 0
            for j in range(i):
                    J[(j, i)] = 0

        for i in range(self.n_hidden):
            h[(i, i)] -= (self.W1[s][i] + self.W2[a][i])

        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if i > j:
                    J[(j, i)] -= self.Wh[j][i]
                if i < j:
                    J[(i, j)] -= self.Wh[j][i]

        for i in range(self.n_hidden):
            if h[(i, i)] == 0:
                del h[(i, i)]
            for j in range(i):
                if J[(j, i)] == 0:
                    del J[(j, i)]
        
        response = SimulatedAnnealingSampler().sample_ising(h, J, num_reads=self.num_reads, num_sweeps=self.num_sweeps, beta_range=self.beta_range)
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

        return h, 0


    def Q(self, s, a):
        h, _ = self.get_h(s, a)
        sum1 = np.dot(self.W1[s], h)
        sum2 = np.dot(self.W2[a], h)
        sum3 = 0
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                sum3 += self.Wh[i][j] * h[i] * h[j]
        sum4 = 0
        for item in h:
            sum4 += item * math.log(item) + (1 - item) * math.log(1 - item)
        result = sum1 + sum2 + sum3 - 1/self.beta * sum4
        return result


    def update_weight(self, delta, state, action):
        h, _ = self.get_h(state, action)
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
        for j in range(self.n_hidden):
            with np.nditer(self.Wh[j], op_flags=['readwrite']) as W_h:
                i = 0
                for w_hh in W_h:
                    w_hh[...] += delta * h[i]
                    i += 1
