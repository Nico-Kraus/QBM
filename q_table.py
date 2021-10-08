class Q_table():
    def __init__(self, n_actions, n_states, params):
        print("use Q-table")
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = params["lr"]
        self.q_table = []
        self.init_q_table(float(params["init"]))

    def init_q_table(self, init):
        for _ in range(self.n_states):
            actions = []
            for _ in range(self.n_actions):
                actions.append(init)
            self.q_table.append(actions)
    
    def Q(self, state, action):
        return self.q_table[state][action]

    def update_weight(self, delta, state, action):
        self.q_table[state][action] += self.lr*delta
