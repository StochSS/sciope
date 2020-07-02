from sciope.utilities.epsilonselectors.epsilon_selector import *
import numpy as np

class RelativeEpsilonSelector(EpsilonSelector):

    def __init__(self, epsilon_percentile, max_rounds = None):
        """
        Parameters
        ----------
        epsilon_percentile : float
            The epsilon percentile to use in each round
        max_rounds : int
            The maximum number of rounds before stopping.  If None, doesn't end.
        """

        self.epsilon_percentile = epsilon_percentile
        self.max_rounds = max_rounds

    def get_initial_epsilon(self):
        """Gets the initial epsilon as the `epsilon_percentile` interpreted
        as a percentile.
        """
        return self.epsilon_percentile, True, self.max_rounds == 0

    def get_epsilon(self, round, abc_history):
        """Computes the next tolerance as the `epsilon_percentile` of the
        previous distances.  Interpreted as an absolute value.
        """

        t = np.percentile(abc_history[-1]['distances'], self.epsilon_percentile)
        return t, False, self.max_rounds and round + 1 == self.max_rounds
