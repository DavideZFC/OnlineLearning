from algos.regret_matching import regret_matching

class blackjack_agent:
    def __init__(self):

        # define dictionary containing one agent for every information set (info sets
        # are the numbers from 1 to 21, to half cards).
        self.agents = {}
        for i in range(1,22):
            # only two actions, since we can only ask another cart or finish
            self.agents[i] = regret_matching(2)

    def pull_arm(self):
        pass