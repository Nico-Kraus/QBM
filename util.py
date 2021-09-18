import numpy as np
import matplotlib.pyplot as plt
import json
import time

def std_dev_from_h(agent):
    print("Testing Annealing params")
    n_hidden = agent.n_hidden
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

    return scores, eps_history


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


def compare_learning_curves(scores, filename):

    x = [i+1 for i in range(len(scores)+1)]

    fig, ax = plt.subplots()

    N = len(scores[0])
    for i in range(len(scores)):
        running_avg = np.empty(N+1)
        color = "C" + str(i)
        name = "Score " + str(i)
        for t in range(N):
            running_avg[t] = np.mean(scores[i][max(0, t-100):(t+1)])
        running_avg[N] = 1
        ax.scatter(x, running_avg, color=color, linewidths=0.5)
        ax.set_ylabel(name, color=color)

    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.set_label_position('left')
    ax.tick_params(axis='y', colors="C0")

    plt.savefig("images/" + filename)


def plot_learning_curve(scores, epsilons, filename):

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
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])
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
    return '%s_%dx%d_h_%d_lr_0,00%d_g_0,%d_b_%d_is_0,%d_an_%d_%d_%d_%d' %(params["method"], env.rows, env.cols, params["n_hidden"], int(params["lr"]*1000), int(params["gamma"]*100), params["beta"], int(params["init_sd"]*10), params["annealing"]["num_reads"], params["annealing"]["num_sweeps"], params["annealing"]["beta_range"][0], params["annealing"]["beta_range"][1])


