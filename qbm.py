import numpy as np
import math
from dwave_qbsolv import QBSolv
from neal.sampler import SimulatedAnnealingSampler

class Qbm():
    def __init__(self, n_actions, n_states, params):
        print("use qbm")
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

        Q = {}

        for i in range(self.n_hidden ):
            Q[(i, i)] = 0
            for j in range(i):
                    Q[(j, i)] = 0

        for i in range(self.n_hidden):
            Q[(i, i)] -= (self.W1[s][i] + self.W2[a][i])

        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if i > j:
                    Q[(j, i)] -= self.Wh[j][i]
                if i < j:
                    Q[(i, j)] -= self.Wh[j][i]

        for i in range(self.n_hidden):
            if Q[(i, i)] == 0:
                del Q[(i, i)]
            for j in range(i):
                if Q[(j, i)] == 0:
                    del Q[(j, i)]
        
        response = SimulatedAnnealingSampler().sample_qubo(Q, num_reads=self.num_reads, num_sweeps=self.num_sweeps, beta_range=self.beta_range)
        samples = list(response.samples())
        h = []
        for i in range(self.n_hidden):
            solutions = []
            for sample in samples:
                solutions.append( sample[i] )
            h_i = np.mean(solutions)
            if h_i >= 1:
                h_i = 0.999999
            if h_i <= 0:
                h_i = 0.000001
            h.append(h_i)

        energies = list(response.record["energy"])
        avg_energie = np.mean(energies)

        return h, avg_energie


    def Q(self, s, a):
        h, avg_energie = self.get_h(s, a)
        sum4 = 0
        for item in h:
            sum4 += item * math.log(item) + (1 - item) * math.log(1 - item)
        result = -avg_energie - 1/self.beta * sum4
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
