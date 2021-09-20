import numpy as np
import os
os.system("")

class Env():

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    def __init__(self, rows=4, cols=4,border="restart"):
        self.map = np.zeros((rows,cols), dtype=int)
        self.start = np.zeros(2, dtype=int) # [row,col] from top left
        self.n_actions = 4
        self.rows = rows
        self.cols = cols
        self.border = border
        self.num_states = self.rows*self.cols
        self.state = np.copy(self.start)


    @staticmethod
    def make_example(num, border):
        if num == 1:
            rows = 2
            cols = 2
            env = Env(rows,cols,border)
            env.set_start(np.zeros(2, dtype=int))
            env.add_reward(rows-1, cols-1)
            env.add_penalty(0,1)
            env.print_env()
            return env
        if num == 2:
            rows = 3
            cols = 3
            env = Env(rows,cols,border)
            env.set_start(np.zeros(2, dtype=int))
            env.add_reward(rows-1, cols-1)
            env.add_penalty(1,2)
            env.add_penalty(2,0)
            env.print_env()
            return env
        if num == 3:
            rows = 4
            cols = 4
            env = Env(rows,cols,border)
            env.set_start(np.zeros(2, dtype=int))
            env.add_reward(rows-1, cols-1)
            env.add_penalty(1,1)
            env.add_penalty(1,3)
            env.add_penalty(2,3)
            env.add_penalty(3,0)
            env.print_env()
            return env
        if num == 4:
            rows = 3
            cols = 5
            env = Env(rows,cols,border)
            start = np.zeros(2, dtype=int)
            start[0] = rows-1
            start[1] = cols-1
            env.set_start(start)
            env.add_reward(0,0)
            env.add_penalty(1,2)
            env.add_penalty(2,2)
            env.print_env()
            return env
        if num == 5:
            rows = 5
            cols = 5
            env = Env(rows,cols,border)
            start = np.zeros(2, dtype=int)
            start[0] = 0
            start[1] = 0
            env.set_start(start)
            env.add_reward(rows-1, cols-1)
            env.add_penalty(1,0)
            env.add_penalty(1,1)
            env.add_penalty(3,1)
            env.add_penalty(0,3)
            env.add_penalty(1,3)
            env.add_penalty(3,3)
            env.add_penalty(3,4)
            env.print_env()
            return env
        if num == 6:
            rows = 8
            cols = 8
            env = Env(rows,cols,border)
            start = np.zeros(2, dtype=int)
            start[0] = 0
            start[1] = 0
            env.set_start(start)
            env.add_reward(rows-1, cols-1)
            # env.add_penalty(2,1)
            # env.add_penalty(1,4)

            #env.add_penalty(5,0)
            env.add_penalty(4,3)
            env.add_penalty(4,4)
            env.add_penalty(4,5)
            #env.add_penalty(7,3)
            #env.add_penalty(0,7)
            env.print_env()
            return env


    def set_start(self, start):
        self.start = start
        self.reset()


    def set_state(self, obs_state):
        self.state[0] = obs_state // self.rows
        self.state[1] = obs_state % self.rows


    def add_reward(self, row, col):
        if row > self.rows or col > self.cols:
            print("reward out of range")
        else:
            self.map[row][col] = 1


    def add_penalty(self, row, col):
        if row > self.rows or col > self.cols:
            print("penalty out of range")
        else:
            self.map[row][col] = -1


    def get_obs_state(self):
        return self.state[1] + self.state[0]*self.cols


    def get_matrix_state(self):
        return self.state


    def get_array_state(self):
        result = [0]*self.num_states
        result[self.state[1] + self.state[0]*self.cols] = 1
        return result


    def take_action(self, action):
        curr_state = np.copy(self.get_matrix_state())
        if action == self.LEFT:
            self.get_matrix_state()[1] -= 1
        if action == self.RIGHT:
            self.get_matrix_state()[1] += 1
        if action == self.DOWN:
            self.get_matrix_state()[0] += 1
        if action == self.UP:
            self.get_matrix_state()[0] -= 1
        row = self.get_matrix_state()[0]
        col = self.get_matrix_state()[1]
        if 0 > row or row > self.rows - 1 or 0 > col or col > self.cols - 1:
            if self.border == "stay":
                self.state = curr_state
            elif self.border == "restart":
                self.get_matrix_state()[0] = 0
                self.get_matrix_state()[1] = 0
        obs = self.get_obs_state()
        reward = self.get_reward()
        if self.map[self.state[0]][self.state[1]] == 0:
            done = False
        else:
            done = True
        return obs, reward, done


    def action_name(self, action):
        if action == 0:
            name = 'LEFT'
        if action == 1:
            name = 'DOWN'
        if action == 2:
            name = 'RIGHT'
        if action == 3:
            name = 'UP'
        return name


    def action_symbol(self, action):
        if action == 0:
            name = '<'
        if action == 1:
            name = 'v'
        if action == 2:
            name = '>'
        if action == 3:
            name = 'n'
        return name


    def get_reward(self):
        if self.map[self.state[0]][self.state[1]] == 1:
            reward = 1
        else:
            reward = 0
        return reward


    def print_env(self):
        for i in range(self.map.shape[0]): # iterate over rows
            for j in range(self.map[i].shape[0]): # iterate over cols
                if self.get_matrix_state()[0] == i and self.get_matrix_state()[1] == j:
                    if self.map[i][j] >= 0:
                        print(' ', end='')
                    print('\x1b[7m' + str(self.map[i][j]) + '\x1b[0m', end='')
                else:
                    if self.map[i][j] >= 0:
                        print(' ', end='')
                    print(self.map[i][j], end='')
            print()
        print()


    def reset(self):
        self.state = np.copy(self.start)
        return self.get_obs_state()