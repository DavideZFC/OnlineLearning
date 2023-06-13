from envs.adversarial_nonconvex_env import adversarial_nonconvex_env
from algos.follow_the_leader import follow_the_leader
from algos.follow_the_perturbed_leader import follow_the_perturbed_leader
import matplotlib.pyplot as plt
import numpy as np

T = 10
N = 200

env = adversarial_nonconvex_env(T,N=N)

policy = follow_the_leader(n_arms=N)

# policy.compute_optimal_lr(T)


for t in range(T):

    action = policy.act()
    print(action)
    loss_vector, loss = env.round(action)
    policy.update(loss_vector)

regret_curve = env.compute_regret_curve()

plt.plot(regret_curve, label='Regret curve')
plt.show()

