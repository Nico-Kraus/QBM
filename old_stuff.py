def play_rdn_sample(env, agent, num_samples):

    scores = []
    for i in range(num_samples):
        obs = env.reset()

        for state in range(env.num_states):
            for action in range(env.n_actions):
                env.set_state(state)
                obs = env.get_obs_state()
                obs_, reward, done = env.take_action(action)
                agent.learn(obs, action, reward, obs_)

        if i % 10 == 0 and i != 0:
            agent.print_policy()
            env.reset()
            score = play(env, agent, 10)
            scores.append(score)
            print('episode: ' + str(i) + ', score: ' + str(score))

        # if i % 100 == 0:
        #     score = play(env, agent, 10, True)
    
    return scores

def play(env, agent, break_loop, print_states=False):

    score = 0
    obs = env.reset()

    for state in range(env.num_states):
        if print_states == True:
            print("play from state " + str(state))
        env.set_state(state)
        index = 0
        done = False
        while not done:
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

# scores1 = play_rdn_sample(env, agent1, num_samples)
# scores2 = play_rdn_sample(env, agent2, num_samples)
# filename = 'rbm_%dx%d_h1_%d_h2_%d_lr1_0,00%d_lr2_0,00%d_g1_0,%d_g2_0,%d_b1_%d_b2_%d_is1_0,%d_is2_0,%d' %(rows, cols, n_hidden1, n_hidden2, int(lr1*1000), int(lr2*1000), int(gamma1*100), int(gamma2*100), beta1, beta2, int(init_sd1*10), int(init_sd2*10))
# x = [i+1 for i in range(len(scores1)+1)]
# compare_learning_curves(x, scores1, scores2, filename)