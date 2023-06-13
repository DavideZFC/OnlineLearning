import numpy as np

class follow_the_leader:
    ''' 
    Hedge algorithm, given a number of arms
    minimizes the regret in the adversarial case
    '''
    def __init__(self, n_arms):
        ''' Initialize the algorithm'''

        self.n_arms = n_arms
        self.weights = np.zeros(n_arms)

    def reset(self):
        self.weights = np.zeros(self.n_arms)

    def act(self):
        ''' Choose the best action from the current weights'''

        x = np.argmin(self.weights)
        return x
    
    def update(self, loss):
        ''' Uses the stored loss to update the weight parameter'''
        
        self.weights += loss