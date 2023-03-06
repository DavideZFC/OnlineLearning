from algos.hedge import hedge
from envs.stochastic_env import stochastic_env
import matplotlib.pyplot as plt
import numpy as np

T = 1000
lr = 0
n_arms = 10

env = stochastic_env(n_arms,T)

policy = hedge(n_arms, lr)
policy.compute_optimal_lr(T)


for t in range(T):

    action = policy.act()
    loss_vector, loss = env.round(action)
    policy.update(loss_vector)

regret_curve = env.compute_regret_curve()

plt.plot(regret_curve, label='Regret curve')
plt.show()

