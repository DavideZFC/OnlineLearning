import numpy as np

class regret_matching:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.x = np.ones(n_arms)/n_arms
        self.r = np.zeros(n_arms)

    def reset(self):
        self.x = np.ones(self.n_arms)/self.n_arms
        self.r = np.zeros(self.n_arms)

    def act(self):
        theta = np.maximum(self.r, np.zeros(self.n_arms))
        if np.sum(theta) == 0:
            self.x = np.ones(self.n_arms)/self.n_arms
            print('zero reached')
        else:
            self.x = theta/np.sum(theta)
        return self.x
    
    def update(self, loss):
        loss = -loss # Quest'algoritmo massimizza l'utilità non minimizza la loss
        forcing_term = np.dot(self.x, loss)*np.ones(self.n_arms)
        self.r = self.r + loss - forcing_term


    