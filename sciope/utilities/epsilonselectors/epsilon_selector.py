from abc import ABCMeta, abstractmethod

class EpsilonSelector(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_initial_epsilon(self):
        """
        Returns
        -------
        epsilon : float
            The initial epsilon value for ABC-SMC
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        has_more : bool
            Whether there are more epsilons after this one
        """

        pass

    @abstractmethod
    def get_epsilon(self, round, abc_history):
        """
        Parameters
        ----------
        round : int
            The round number
        abc_results :
            A list of dictionaries with keys `accepted_samples` and `distances`
            representing the history of all ABC runs up to this point

        Returns
        -------
        epsilon : float
            The epsilon value for ABC-SMC
        percentile : bool
            Whether the epsilon should be interpreted as a percentile
        terminate : bool
            Whether to stop after this epsilon
        """
        pass
