import numpy as np
import matplotlib.pyplot as plt
from algos.hedge import hedge
from algos.ogd import gradient_descent
from algos.regret_matching import regret_matching
from envs.two_adversary_env import adversarial_env
from misc.test_adversarial_env import test_adversarial_env
from misc.confidence_bounds import bootstrap_ci
from misc.plot_from_dataset import plot_data


T = 1000
lr = 0
seeds = 10
x = np.arange(T)

n_arms_1 = 2
n_arms_2 = 2

####
# Define env
####

game_matrix = np.array([[1.,0],[0,-1]])
env = adversarial_env(game_matrix, T)

####
# Define players
####

player1 = hedge(n_arms_1, lr)
label1 = 'Hedge'
# player2 = regret_matching(n_arms_2)
player2 = gradient_descent(n_arms_2, lr)
label2 = 'GD'

player1.compute_optimal_lr(T)
player2.compute_optimal_lr(T)

####
# Make the test
####

regret_matrix_1, regret_matrix_2 = test_adversarial_env(env, player1, player2, T, seeds)
low1, high1 = bootstrap_ci(regret_matrix_1)
plot_data(x, low1, high1, col='C0', label=label1)
low2, high2 = bootstrap_ci(regret_matrix_2)
plot_data(x, low2, high2, col='C1', label=label2)

####
# Plot and save
####


plt.legend()
plt.savefig('results/{}{}.pdf'.format(label1,label2))
plt.show()
