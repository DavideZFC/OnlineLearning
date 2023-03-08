from algos.hedge import hedge
from algos.ogd import gradient_descent
from envs.stochastic_env import stochastic_env
from misc.confidence_bounds import bootstrap_ci
from misc.plot_from_dataset import plot_data
import matplotlib.pyplot as plt
import numpy as np

T = 5000
lr = 0
n_arms = 10



policy_gd = gradient_descent(n_arms, lr) 
policy_hedge = hedge(n_arms, lr)

policy_gd.compute_optimal_lr(T)
policy_hedge.compute_optimal_lr(T)

def test_policy(policy, T, N):
    env = stochastic_env(n_arms,T)
    results = np.zeros((N,T))

    for i in range(N):
        policy.reset()
        env.reset()

        for t in range(T):
            action = policy.act()
            loss_vector, loss = env.round(action)
            policy.update(loss_vector)

        regret_curve = env.compute_regret_curve()
        results[i,:] = regret_curve

    return results


N = 50
results_gd = test_policy(policy_gd, T, N)
results_hedge = test_policy(policy_hedge, T, N)

low_gd, high_gd = bootstrap_ci(results_gd)
low_hedge, high_edge = bootstrap_ci(results_hedge)

x = np.linspace(1,T,T)

plot_data(x, low_gd, high_gd, col='C1', label='Gradient descent')
plot_data(x, low_hedge, high_edge, col='C2', label='Hedge')

plt.legend()
plt.title('Regret of Hedge vs Gradient descent')
plt.show()

