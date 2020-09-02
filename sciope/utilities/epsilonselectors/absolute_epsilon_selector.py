"""
Absolute epsilon selector
"""

from sciope.utilities.epsilonselectors.epsilon_selector import *

class AbsoluteEpsilonSelector(EpsilonSelector):
    """
    Creates an epsilon selector based on a fixed sequence.
    """

    def __init__(self, epsilon_sequence):
        """

        Parameters
        ----------
        epsilon_sequence : Seqeunce[float]
            Sequence of epsilons to use.
        """

        assert(len(epsilon_sequence) > 0)
        self.epsilon_sequence = epsilon_sequence
        self.last_round = len(self.epsilon_sequence) - 1

    def get_initial_epsilon(self):
        """Gets the first epsilon in the sequence.

        Returns
        -------
        epsilon : float
            The initial epsilon value of this sequence
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        has_more : bool
            Whether there are more epsilons after this one
        """
        return self.epsilon_sequence[0], False, len(self.epsilon_sequence) == 1

    def get_epsilon(self, round, abc_history):
        """Returns the n-th epsilon in the seqeunce.

        Parameters
        ----------
        round : int
            the round to get the epsilon for
        abc_history : type
            A list of dictionaries with keys `accepted_samples` and `distances`
            representing the history of all ABC runs up to this point.

        Returns
        -------
        epsilon : float
            The epsilon value for ABC-SMC
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        terminate : bool
            Whether to stop after this epsilon
        """
        return self.epsilon_sequence[round], False, round == last_round
