import numpy as np

class gradient_descent:
    def __init__(self, n_arms, learning_rate):
        ''' Initialize the algorithm'''

        self.n_arms = n_arms
        self.x = np.ones(n_arms)/n_arms
        self.lr = learning_rate

    def reset(self):
        self.x = np.ones(self.n_arms)/self.n_arms

    def compute_optimal_lr(self,T):
        ''' Using information about the simplex and the time horizon, compute the learning rate'''

        # diameter of n-dimensional simplex
        D = 2**0.5 

        # upper bound for the gradient
        G = self.n_arms**0.5

        # leatning rate
        self.lr = D/(G*T**0.5)
        print('## Optimal learning rate computed: '+str(self.lr))

    def act(self):
        ''' Chooose the action. In this case, the action is just the hidden parameter'''

        return self.x
    
    def proj(self,x):
        ''' 
        Projector allows to avoid that the gradient update 
        goes out from the simplex
        '''

        x = np.maximum(x,0)
        x = x/np.sum(x)
        return x
    
    def update(self, loss):
        '''
        Gradient descent update
        '''
        
        self.x = self.proj(self.x-self.lr*loss)