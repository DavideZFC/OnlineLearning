import numpy as np

class regret_matching:
    def __init__(self, n_arms):
        ''' Initialize the algorithm'''

        self.n_arms = n_arms
        self.x = np.ones(n_arms)/n_arms
        self.r = np.zeros(n_arms)

    def reset(self):
        # current action
        self.x = np.ones(self.n_arms)/self.n_arms

        # vector of past rewards
        self.r = np.zeros(self.n_arms)

    def act(self):
        " Chooses current action x basing on the r vector"

        # get the positive part from the r vector
        theta = np.maximum(self.r, np.zeros(self.n_arms))

        if np.sum(theta) == 0:
            self.x = np.ones(self.n_arms)/self.n_arms
        else:
            # x is a normalized version of r
            self.x = theta/np.sum(theta)
        return self.x
    
    def update(self, loss):
        """ Update r vector"""

        # this algorithm is designed to maximize utility, so we have to do this passage
        loss = -loss

        # generate constant vector proportinal to the loss experienced
        forcing_term = np.dot(self.x, loss)*np.ones(self.n_arms)

        # update r with loss and forcing term
        self.r = self.r + loss - forcing_term


    