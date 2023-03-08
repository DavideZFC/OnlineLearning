import numpy as np

class hedge:
    def __init__(self, n_arms, learning_rate):

        self.n_arms = n_arms
        self.weights = np.ones(n_arms)
        self.lr = learning_rate

    def reset(self):
        self.weights = np.ones(self.n_arms)

    def compute_optimal_lr(self,T):
        self.lr = (np.log(self.n_arms)/T)**0.5
        print('## Optimal learning rate computed: '+str(self.lr))

    def act(self):
        x = self.weights/np.sum(self.weights)
        return x
    
    def update(self, loss):
        for i in range(self.n_arms):
            coef = np.exp(-self.lr*loss[i])
            self.weights[i] = coef*self.weights[i]
