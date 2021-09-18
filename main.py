from env import Env
from agent import Agent
from util import load_config, create_filename, create_filename_compare, compare_learning_curves, plot_learning_curve, play_eps_greedy_rounds, play_eps_greedy

if __name__ == '__main__':

    params = load_config("params.json")

    env = Env.make_example(params["env"]["example"], params["env"]["border_rule"])

    # agent = Agent(env, params["agents"][0])
    # scores, eps_history = play_eps_greedy_rounds(env, agent, params["play"]["num_samples"], params["play"]["max_steps"], [[3,2],[1,2],[0,0]])

    compare_scores = []
    for agent_params in params["agents"]:
        agent = Agent(env, agent_params)
        scores, eps_history = play_eps_greedy(env, agent, params["play"]["num_samples"], params["play"]["max_steps"], env.start)
        compare_scores.append(scores)

    filename = create_filename_compare(env, params["agents"])
    compare_learning_curves(compare_scores, filename)