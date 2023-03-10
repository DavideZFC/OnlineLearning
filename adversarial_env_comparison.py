import numpy as np
import matplotlib.pyplot as plt
from algos.hedge import hedge
from algos.ogd import gradient_descent
from envs.two_adversary_env import adversarial_env


T = 10000
lr = 0

n_arms_1 = 2
n_arms_2 = 2
# game_matrix = np.random.uniform(0,1,size=(n_arms_1, n_arms_2))
game_matrix = np.array([[1.,0],[0,-1]])
env = adversarial_env(game_matrix, T)

print('This is the game matrix:')
print(game_matrix)
print('Player1: maximizer - - - - Player2: minimizer')

player1 = hedge(n_arms_1, lr)
player2 = gradient_descent(n_arms_2, lr)

player1.compute_optimal_lr(T)
player2.compute_optimal_lr(T)

player_1_strategies = np.zeros((T,n_arms_1))
player_2_strategies = np.zeros((T,n_arms_2))

for t in range(T):

    action1 = player1.act()
    action2 = player2.act()

    player_1_strategies[t,:] = action1
    player_2_strategies[t,:] = action2

    ret_1, loss_1, ret_2, loss_2 = env.round(action1,action2)
    player1.update(ret_1)
    player2.update(ret_2)

regret_curve_1, regret_curve_2 = env.compute_regret_curves()

print('Average strategy player 1:')
print(np.mean(player_1_strategies,axis=0))
print('Average strategy player 2:')
print(np.mean(player_2_strategies,axis=0))

plt.plot(regret_curve_1, label='Regret curve Player 1')
plt.plot(regret_curve_2, label='Regret curve Player 2')
plt.legend()
plt.show()