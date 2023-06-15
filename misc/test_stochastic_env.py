import numpy as np


def test_policy(env, policy, T, seeds=1):
    # test the policy on given environment for a number of seeds spacified
    regret_matrix = np.zeros((seeds, T))

    for s in range(seeds):
        np.random.seed(s)
        policy.reset()
        env.reset()

        for t in range(T):

            action = policy.act()
            loss_vector, loss = env.round(action)
            policy.update(loss_vector)

        regret_matrix[s,:] = env.compute_regret_curve()

    return regret_matrix