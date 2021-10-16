import numpy as np
import math
from dwave.system import DWaveSampler, EmbeddingComposite

class Qabm():
    def __init__(self, n_actions, n_states, params):
        print("use qabm")
        self.lr = params["lr"]
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_hidden = params["n_hidden"]
        self.num_reads = params["annealing"]["num_reads"]
        self.annealing_time = params["annealing"]["annealing_time"]
        self.W1 = np.random.normal(0, params["init_sd"], (self.n_states, self.n_hidden))
        self.W2 = np.random.normal(0, params["init_sd"], (self.n_actions, self.n_hidden))
        self.Wh = np.random.normal(0, params["init_sd"], (self.n_hidden, self.n_hidden))
        self.sampler = EmbeddingComposite(DWaveSampler())


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
        
        response = self.sampler.sample_qubo(Q, num_reads=self.num_reads, annealing_time=self.annealing_time)
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

        return h, 0


    def Q(self, s, a):
        h, _ = self.get_h(s, a)
        sum1 = np.dot(self.W1[s], h)
        sum2 = np.dot(self.W2[a], h)
        sum3 = 0
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                sum3 += self.Wh[i][j] * h[i] * h[j]
        result = sum1 + sum2 + sum3
        return result


    def update_weight(self, delta, state, action):
        delta *= self.lr
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
