import numpy as np

class follow_the_perturbed_leader:
    ''' 
    Hedge algorithm, given a number of arms
    minimizes the regret in the adversarial case
    '''
    def __init__(self, n_arms, eta):
        ''' Initialize the algorithm'''

        self.n_arms = n_arms
        self.eta = eta

        self.weights = -np.random.exponential(self.eta)*np.linspace(-1,1,self.n_arms)

    def reset(self):
        self.weights = -np.random.exponential(self.eta)*np.linspace(-1,1,self.n_arms)

    def act(self):
        ''' Choose the best action from the current weights'''

        x = np.argmin(self.weights)
        return x
    
    def update(self, loss):
        ''' Uses the stored loss to update the weight parameter'''
        
        self.weights += loss