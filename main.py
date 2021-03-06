from env import Env
from agent import Agent
from util import *

if __name__ == '__main__':

    params = load_config("params.json")

    env = Env.make_example(params["env"]["example"], params["env"]["border_rule"])

    compare_scores = []
    for agent_params in params["agents"]:
        agent = Agent(env, agent_params)
        scores, eps_history = play_eps_greedy(env, agent, params["play"]["num_samples"], params["play"]["max_steps"], env.start)
        # scores, eps_history = play_eps_greedy_rounds(env, agent, params["play"]["num_samples"], params["play"]["max_steps"], [[2,0],[8,0],[6,8],[0,10],[0,0]])
        # scores = play_rdn_sample(env, agent, params["play"]["num_samples"], params["play"]["max_steps"], env.start)
        compare_scores.append(scores)

    filename = create_filename_compare(env, params["agents"])
    # plot_learning_curve(scores, eps_history, filename, avg = 100)
    compare_learning_curves(compare_scores, filename, params["agents"], avg = 100)
    # compare_learning_curves_rdn(compare_scores, filename, params["agents"], avg = 1)