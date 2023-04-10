from algos.hedge import hedge
from algos.ogd import gradient_descent
from algos.regret_matching import regret_matching
from envs.stochastic_env import stochastic_env
import matplotlib.pyplot as plt
import numpy as np
from envs.bilevel_stochastic_env import bilevel_stochastic_env

env = bilevel_stochastic_env(n_arms_first=2, n_arms_second=3, T=100)

print(env.get_losses())