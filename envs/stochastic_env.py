import numpy as np

class stochastic_env:

    def __init__(self, n_arms, T, means=[]):
        
        self.T = T
        self.n_arms = n_arms
        if means == []:
            self.means = np.linspace(0.4,0.6,n_arms)
        else:
            self.means = np.copy(means)

        self.loss_history = np.zeros(T)
        self.ret_history = np.zeros((T,n_arms))
        self.t = 0

    def reset(self):
        self.loss_history = np.zeros(self.T)
        self.ret_history = np.zeros((self.T, self.n_arms))
        self.t = 0

    def get_loss(self):
        ''' returns the loss vector without pulling any action'''

        ret = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            ret[i] = np.random.binomial(n=1,p=self.means[i])

        self.t += 1

        return ret


    def round(self, action):

        ret = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            ret[i] = np.random.binomial(n=1,p=self.means[i])
        loss = np.dot(ret, action)

        self.loss_history[self.t] = loss
        self.ret_history[self.t,:] = ret

        self.t += 1

        return ret, loss
    
    def compute_regret_curve(self):

        arm_performances = np.sum(self.ret_history, axis=0)
        best_arm = np.argmin(arm_performances)

        return np.cumsum(self.loss_history-self.ret_history[:,best_arm])

        