class Q_table():
    def __init__(self, n_actions, n_states):
        print("use Q-table")
        self.n_actions = n_actions
        self.n_states = n_states
        self.q_table = []
        self.init_q_table()

    def init_q_table(self):
        for _ in range(self.n_states):
            actions = []
            for _ in range(self.n_actions):
                actions.append(0.0)
            self.q_table.append(actions)
    
    def Q(self, state, action):
        return self.q_table[state][action]

    def update_weight(self, delta, state, action):
        self.q_table[state][action] += delta
