from env import Env
from agent import Agent
from util import load_config, create_filename, plot_learning_curve, play_eps_greedy_rounds, play_eps_greedy, std_dev_from_h

if __name__ == '__main__':

    env = Env.make_example(1, "stay")

    params = load_config("params.json")
    agent = Agent(env, **params)

    # std_dev_from_h(agent)

    num_samples = 10000
    loop_break = 4
    # scores,eps_history = play_eps_greedy_rounds(env, agent, num_samples, loop_break, [[0,4],[2,4]])
    scores,eps_history = play_eps_greedy(env, agent, num_samples, loop_break, env.start)

    filename = create_filename(env, params)
    plot_learning_curve(scores, eps_history, filename)