import numpy as np

class hedge:
    ''' 
    Hedge algorithm, given a number of arms
    minimizes the regret in the adversarial case
    '''
    def __init__(self, n_arms, learning_rate=1.0):
        ''' Initialize the algorithm'''

        self.n_arms = n_arms
        self.weights = np.ones(n_arms)
        self.lr = learning_rate

    def reset(self):
        self.weights = np.ones(self.n_arms)

    def compute_optimal_lr(self,T):
        '''
        Computes the optimal learning rate for the algorithm
        from the time horizon
        '''

        self.lr = (np.log(self.n_arms)/T)**0.5
        print('## Optimal learning rate computed: '+str(self.lr))

    def act(self):
        ''' Choose the best action from the current weights'''

        x = self.weights/np.sum(self.weights)
        return x
    
    def update(self, loss):
        ''' Uses the stored loss to update the weight parameter'''
        
        for i in range(self.n_arms):
            coef = np.exp(-self.lr*loss[i])
            self.weights[i] = coef*self.weights[i]
