import numpy as np


def test_adversarial_env(env, player1, player2, T, seeds=1):
    # test the policy on given environment for a number of seeds spacified
    regret_matrix_1 = np.zeros((seeds, T))
    regret_matrix_2 = np.zeros((seeds, T))

    for s in range(seeds):
        np.random.seed(s)
        player1.reset()
        player2.reset()
        env.reset()

        for t in range(T):

            action1 = player1.act()
            action2 = player2.act()

            ret_1, loss_1, ret_2, loss_2 = env.round(action1,action2)
            player1.update(ret_1)
            player2.update(ret_2)

        regret_curve_1, regret_curve_2 = env.compute_regret_curves()

        regret_matrix_1[s,:] = regret_curve_1
        regret_matrix_2[s,:] = regret_curve_2

    return regret_matrix_1, regret_matrix_2