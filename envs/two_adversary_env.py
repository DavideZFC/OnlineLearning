import numpy as np

class adversarial_env:

    def __init__(self, matrix, T):
        self.n_arms_1, self.n_arms_2 = matrix.shape
        self.mat = np.copy(matrix)

        self.loss_history_first = np.zeros(T)
        self.ret_history_first = np.zeros((T,self.n_arms_1))

        self.loss_history_second = np.zeros(T)
        self.ret_history_second = np.zeros((T,self.n_arms_2))
        self.t = 0

    def round(self, action_1, action_2):
        ret_1 = -np.dot(self.mat, action_2)
        ret_2 = np.dot(action_1, self.mat)

        loss_1 = np.dot(ret_1, action_1)
        loss_2 = np.dot(ret_2, action_2)

        self.loss_history_first[self.t] = loss_1
        self.ret_history_first[self.t,:] = ret_1

        self.loss_history_second[self.t] = loss_2
        self.ret_history_second[self.t,:] = ret_2

        self.t += 1

        return ret_1, loss_1, ret_2, loss_2

    def compute_regret_curves(self):

        first_performances = np.sum(self.ret_history_first, axis=0)
        best_arm_1 = np.argmin(first_performances)

        second_performances = np.sum(self.ret_history_second, axis=0)
        best_arm_2 = np.argmin(second_performances)

        return np.cumsum(self.loss_history_first-self.ret_history_first[:,best_arm_1]), np.cumsum(self.loss_history_second-self.ret_history_second[:,best_arm_2])      




