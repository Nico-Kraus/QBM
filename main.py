from env import Env
from agent import Agent
from util import load_config, create_filename, plot_learning_curve, play_eps_greedy_rounds, play_eps_greedy, std_dev_from_h

if __name__ == '__main__':

    params = load_config("params.json")

    env = Env.make_example(params["env"]["example"], "stay")
    agent = Agent(env, params["agent"])

    std_dev_from_h(agent)

    # scores,eps_history = play_eps_greedy_rounds(env, agent, num_samples, loop_break, [[0,4],[2,4]])
    scores,eps_history = play_eps_greedy(env, agent, params["play"]["num_samples"],params["play"]["max_steps"], env.start)

    filename = create_filename(env, params["agent"])
    plot_learning_curve(scores, eps_history, filename)