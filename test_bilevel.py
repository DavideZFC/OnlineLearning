import matplotlib.pyplot as plt
import numpy as np
from envs.bilevel_stochastic_env import bilevel_stochastic_env
from algos.convex_learner import convex_learner

n_arms_first = 2
n_arms_second = 3
T = 1000
N_exp = 50



regret_vec = np.zeros(T)
for i in range(N_exp):
    env = bilevel_stochastic_env(n_arms_first , n_arms_second, T)
    policy = convex_learner(n_arms_first , n_arms_second)
    for t in range(T):
        ret = policy.update(env.get_losses())
        env.store_rewards(ret)
    regret_vec += env.compute_regret_curve()

plt.plot(regret_vec/N_exp)
plt.show()