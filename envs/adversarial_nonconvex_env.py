import numpy as np

class adversarial_nonconvex_env:

    def __init__(self, T, N=1000):
        
        self.T = T
        self.N = N
        self.arms = np.linspace(-1,1,N)

        self.loss_history = np.zeros(T)
        self.ret_history = np.zeros((T,N))
        self.t = 0

    def reset(self):
        self.loss_history = np.zeros(self.T)
        self.ret_history = np.zeros((self.T, self.n_arms))
        self.t = 0

    def sigmoid(self, x):
        return np.exp(x)/(np.exp(x)+np.exp(-x))

    def get_loss(self):
        ''' returns the loss vector without pulling any action'''

        ret = self.sigmoid((-1)**self.t*self.arms)

        if (self.t == 0):
            ret /= 2

        self.t += 1

        return ret


    def round(self, action):


        ret = self.sigmoid((-1)**self.t*self.arms)

        if (self.t == 0):
            ret /= 2

        self.ret_history[self.t, :] = ret
        self.loss_history[self.t] = ret[action]
        self.t += 1

        return ret, ret[action]
    
    def compute_regret_curve(self):

        arm_performances = np.sum(self.ret_history, axis=0)
        best_arm = np.argmin(arm_performances)

        return np.cumsum(self.loss_history-self.ret_history[:,best_arm])