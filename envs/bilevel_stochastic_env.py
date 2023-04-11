import numpy as np
from envs.stochastic_env import stochastic_env

class bilevel_stochastic_env:

    def __init__(self, n_arms_first, n_arms_second, T, means=[]):

        self.n_arms_first = n_arms_first
        self.n_arms_second = n_arms_second

        # time counter
        self.t = 0
        self.reward_vector = np.zeros(T)

        self.base_envs = []

        if means == []:
            for i in range(n_arms_first):                
                means.append(np.linspace(0.4*i+(1-i),0.6*i+(1-i),self.n_arms_second))
        
        self.optimal_arm = 1000

        for i in range(n_arms_first):
            # store all the second level envs in this list
            self.base_envs.append(stochastic_env(n_arms_second, T=T, means = means[i]))

            # compute optimal arm in the multi stage process
            self.optimal_arm = min(self.optimal_arm, np.min(means[i]))
        
        print('Best arm value: {}'.format(self.optimal_arm))


    def get_losses(self):

        losses = []
        for i in range(self.n_arms_first):
            losses.append(self.base_envs[i].get_loss())

        return losses
    
    def store_rewards(self, reward):
        ''' with this function, we store in a vector the rewards gained'''
        self.reward_vector[self.t] = reward
        self.t += 1

    
    def compute_regret_curve(self):
        return np.cumsum(self.reward_vector-self.optimal_arm)


        