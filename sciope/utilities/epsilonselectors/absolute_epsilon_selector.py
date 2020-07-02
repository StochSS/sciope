from sciope.utilities.epsilonselectors.epsilon_selector import *

class AbsoluteEpsilonSelector(EpsilonSelector):

    def __init__(self, epsilon_sequence):

        assert(len(epsilon_sequence) > 0)
        self.epsilon_sequence = epsilon_sequence
        self.last_round = len(self.epsilon_sequence) - 1

    def get_initial_epsilon(self):
        return self.epsilon_sequence[0], False, len(self.epsilon_sequence) == 1

    def get_epsilon(self, round, abc_history):
        return self.epsilon_sequence[round], False, round == last_round
