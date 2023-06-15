from algos.hedge import hedge
from algos.ogd import gradient_descent
from algos.regret_matching import regret_matching
from envs.stochastic_env import stochastic_env
from misc.confidence_bounds import bootstrap_ci
from misc.plot_from_dataset import plot_data
from misc.test_stochastic_env import test_policy
import matplotlib.pyplot as plt
import numpy as np

T = 500
lr = 0
n_arms = 10
seeds = 5
x = np.arange(T)

env = stochastic_env(n_arms=n_arms, T=T)

####
# define the policies to use
####

p1 = gradient_descent(n_arms, lr)
p1.compute_optimal_lr(T)

p2 = hedge(n_arms, lr)
p2.compute_optimal_lr(T)

p3 = regret_matching(n_arms)

####
# select policies for this experiment
####

policies = [ p1, p2, p3]
labels = ['GD', 'Hedge', 'RM']


savename = ''
for i in range(len(policies)):
    regret_matrix = test_policy(env, policies[i], T, seeds)
    low, high = bootstrap_ci(regret_matrix)
    plot_data(x, low, high, col='C{}'.format(i), label=labels[i])
    savename += labels[i]


plt.legend()
plt.savefig('results/{}.pdf'.format(savename))
plt.show()

