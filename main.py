from env import Env
from agent import Agent
from util import load_config, create_filename, plot_learning_curve, play_eps_greedy_rounds, play_eps_greedy

if __name__ == '__main__':

    params = load_config("params.json")

    env = Env.make_example(params["env"]["example"], "stay")
    agent = Agent(env, params["agent"])

    # scores,eps_history = play_eps_greedy_rounds(env, agent, params["play"]["num_samples"], params["play"]["max_steps"], [[3,2],[1,2],[0,0]])
    scores,eps_history = play_eps_greedy(env, agent, params["play"]["num_samples"], params["play"]["max_steps"], env.start)

    filename = create_filename(env, params["agent"])
    plot_learning_curve(scores, eps_history, filename)