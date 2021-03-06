import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def std_dev_from_h(agent):
    print("Testing Annealing params")
    n_hidden = agent.method.n_hidden
    all_values = [[] for _ in range(n_hidden)]
    
    start = time.time()
    
    for _ in range(100):
        h, _ = agent.method.get_h(0,0)
        for i in range(n_hidden):
            all_values[i].append(h[i])
    
    end = time.time()
    print(str(end - start) + " Seconds")

    stds = []
    for values in all_values:
        std = np.std(values)
        stds.append(std)
    mean = np.mean(stds)
    h, _ = agent.method.get_h(0,0)
    print(h)
    print(str(mean) +  " Mean std")


def play_eps_greedy(env, agent, num_samples, max_steps, start):

    env.set_start(start)
    agent.print_policy()

    scores = []
    eps_history = []
    for i in range(num_samples):
        score = 0
        done = False
        obs = env.reset()

        index = 0
        done = False
        while not done:
            action, epsilon = agent.choose_eps_greedy_action(obs, num_samples*0.9, i)
            obs_, reward, done = env.take_action(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
            index += 1
            if index >= max_steps:
                break
        
        print('.', end='', flush=True)
        scores.append(score)
        eps_history.append(epsilon)

        if i % 100 == 0 and i != 0:
            print()
            agent.print_policy()
            avg_score = np.mean(scores[-100:])
            print('episode: %d, avg score: %.2f, epsilon: %.2f' %(i, avg_score, epsilon))

    print()

    return scores, eps_history

def play_rounds(env, agent, num_samples, max_steps, start):

    env.set_start(start)
    agent.print_policy()

    scores = []
    for i in range(num_samples):
        score = 0
        done = False
        obs = env.reset()

        index = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done = env.take_action(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
            index += 1
            if index >= max_steps:
                break
        
        print('.', end='', flush=True)
        scores.append(score)

        if i % 100 == 0 and i != 0:
            print()
            agent.print_policy()
            avg_score = np.mean(scores[-100:])
            print('episode: %d, avg score: %.2f' %(i, avg_score))

    print()

    return scores


def play_eps_greedy_rounds(env, agent, num_samples, max_steps, fields):
    
    scores = []
    eps_history = []
    for field in fields:
        start = np.zeros(2, dtype=int)
        start[0] = field[0]
        start[1] = field[1]
        new_scores, new_eps_history = play_eps_greedy(env, agent, num_samples, max_steps, start)
        scores += new_scores
        eps_history += new_eps_history
    return scores, eps_history


def play_rdn_sample(env, agent, num_samples, max_steps, fields):

    scores = []
    for i in range(num_samples):
        obs = env.reset()

        for state in range(env.num_states):
            env.set_state(state)
            matrix_state = env.get_matrix_state()
            if env.map[matrix_state[0]][matrix_state[1]] == 0:
                for action in range(env.n_actions):
                    env.set_state(state)
                    obs = env.get_obs_state()
                    obs_, reward, done = env.take_action(action)
                    agent.learn(obs, action, reward, obs_)

        if i % 10 == 0 and i != 0:
            env.reset()
            score = play(env, agent, max_steps, False)
            agent.print_policy()
            scores.append(score)
            print('episode: ' + str(i) + ', score: ' + str(score))
    
    return scores


def play(env, agent, break_loop, print_states=False):

    score = 0
    obs = env.reset()

    for state in range(env.num_states):
        if print_states == True:
            print("play from state " + str(state))
        env.set_state(state)
        matrix_state = env.get_matrix_state()
        if env.map[matrix_state[0]][matrix_state[1]] == 0:
            if print_states == True:
                env.print_env()
            index = 0
            done = False
            while not done:
                obs = env.get_obs_state()
                action = agent.choose_action(obs)
                obs_, reward, done = env.take_action(action)
                if print_states == True:
                    env.print_env()
                score += reward
                obs = obs_
                index += 1
                if index > break_loop:
                    break

    return score


def compare_learning_curves(scores, filename, agents, avg):

    x = [i+1 for i in range(len(scores[0])+1)]

    fig, ax = plt.subplots()

    N = len(scores[0])
    for i in range(len(scores)):
        running_avg = np.empty(N+1)
        color = "C" + str(i)
        for t in range(N):
            running_avg[t] = np.mean(scores[i][max(0, t-avg):(t+1)])
        running_avg[N] = 1
        ax.plot(x, running_avg, color=color, linewidth=1,label=agents[i]["name"])
        
    ax.set_ylabel("Score", color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.legend(loc='upper left', borderaxespad=0.)
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.set_label_position('left')
    ax.tick_params(axis='y', colors="C0")
    ax.tick_params(axis='x', colors="C0")

    plt.savefig("images/" + filename, dpi=300)


def compare_learning_curves_rdn(scores, filename, agents, avg):

    x = [i+1 for i in range(len(scores[0])+1)]

    fig, ax = plt.subplots()

    max_score = 0
    for score in scores:
        new_max = max(score)
        if new_max > max_score:
            max_score = new_max

    N = len(scores[0])
    for i in range(len(scores)):
        running_avg = np.empty(N+1)
        color = "C" + str(i)
        for t in range(N):
            running_avg[t] = np.mean(scores[i][max(0, t-avg):(t+1)])
        running_avg[N] = max([1, max_score])
        ax.plot(x, running_avg, color=color, linewidth=1,label=agents[i]["name"])
        
    ax.set_ylabel("Score", color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.legend(loc='upper left', borderaxespad=0.)
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.set_label_position('left')
    ax.tick_params(axis='y', colors="C0")
    ax.tick_params(axis='x', colors="C0")

    plt.savefig("images/" + filename, dpi=300)


def plot_learning_curve(scores, epsilons, filename, avg):

    x = [i+1 for i in range(len(scores)+1)]
    
    fig, ax = plt.subplots()
    epsilons.append(0)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N+1)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-avg):(t+1)])
    running_avg[N] = 1

    ax2 = ax.twinx()

    ax2.plot(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig("images/" + filename)


def load_config(path_to_config):
    with open(path_to_config, "r") as config_json:
        config = json.load(config_json)
    return config


def create_filename(env, params):
    return '%s_%dx%d_h_%d_lr_0,00%d_g_0,%d_b_%d_is_0,%d_an_%d_%d_%d_%d' %(params["method"], env.rows, env.cols, params["n_hidden"], int(params["lr"]*1000), params["gamma"]*100, int(params["beta"]), int(params["init_sd"]*10), params["annealing"]["num_reads"], params["annealing"]["num_sweeps"], params["annealing"]["beta_range"][0], params["annealing"]["beta_range"][1])


def create_filename_compare(env, agents):
    filename = '%dx%d' %(env.rows, env.cols)
    for agent in agents:
        filename += '_%s' %(agent["name"])
    return filename


