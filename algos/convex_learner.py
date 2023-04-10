from algos.regret_matching import regret_matching
import numpy as np

class convex_learner:
    ''' this class contains an optimizer for a two phases problem, where first
    we choose how to weight some learners, and then we receive the loss from each of them'''
    def __init__(self, n_arms_first, n_arms_second):

        self.n_arms_first = n_arms_first
        self.n_arms_second = n_arms_second


        self.base_agent = regret_matching(n_arms_first)

        self.agents = []
        for i in range(n_arms_first):
            self.agents[i] = regret_matching(n_arms_second)

    def act(self):
        pass

    def update(self, losses):

        main_loss = np.zeros(self.n_arms_first)

        for i in range(self.n_arms_first):

            # action the i-th learner would have done
            action = self.agents[i].act()

            # update second level learners
            self.agents[i].update(losses[i])

            # compute loss for first level learner
            main_loss[i] = np.dot(action, losses[i])

        # update base learner
        self.base_agent.update(main_loss)

            


        