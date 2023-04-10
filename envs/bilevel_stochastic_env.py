import numpy as np
from envs.stochastic_env import stochastic_env

class bilevel_stochastic_env:

    def __init__(self, n_arms_first, n_arms_second, T, means=[]):

        self.n_arms_first = n_arms_first
        self.n_arms_second = n_arms_second

        self.base_envs = []

        if means == []:
            for i in range(n_arms_first):                
                means.append(np.linspace(0.4*i+(1-i),0.6*i+(1-i),self.n_arms_second))
        
        for i in range(n_arms_first):
            self.base_envs.append(stochastic_env(n_arms_second, T=T, means = means[i]))


    def get_losses(self):

        losses = []
        for i in range(self.n_arms_first):
            losses.append(self.base_envs[i].get_loss())

        return losses


    def round(self, action):

        pass
    
    def compute_regret_curve(self):

        pass

        