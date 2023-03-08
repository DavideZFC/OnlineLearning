import numpy as np

class gradient_descent:
    def __init__(self, n_arms, learning_rate):

        self.n_arms = n_arms
        self.x = np.ones(n_arms)/n_arms
        self.lr = learning_rate

    def reset(self):
        self.x = np.ones(self.n_arms)/self.n_arms

    def compute_optimal_lr(self,T):
        D = 2**0.5 # diameter of n-dimensional simplex
        G = self.n_arms**0.5
        self.lr = D/(G*T**0.5)
        print('## Optimal learning rate computed: '+str(self.lr))

    def act(self):
        return self.x
    
    def proj(self,x):
        x = np.maximum(x,0)
        x = x/np.sum(x)
        return x
    
    def update(self, loss):
        self.x = self.proj(self.x-self.lr*loss)