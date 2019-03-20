from mio.utilities.summarystats.summary_base import SummaryBase
from mio.features.feature_extraction import generate_tsfresh_features
from mio.designs.random_sampling import RandomSampling
from tsfresh.feature_extraction import MinimalFCParameters
from mio.data.dataset import DataSet

class SummariesTSFRESH(SummaryBase):
    """
    An ensemble of different statistics from TSFRESH
    """

    def __init__(self):
        self.name = 'SummariesTSFRESH'
        self.features = None
        super(SummariesTSFRESH, self).__init__(self.name)

    def compute(self, data, features=MinimalFCParameters()):
        self.features = features
        num_species = data.shape[2]
        num_points = data.shape[0]
        
        #here we aggregate features from several species into one feature vector
        feature_values = []
        for i in range(num_species):
            feature_values.append(generate_tsfresh_features(data[:,:,i], features))
        
        # ToDo: Check for NaNs
        return feature_values

class DataSetMET(DataSet):

    def __init__(self):
        name = 'stochmet'
        super(FactorialDesign, self).__init__(name)
        self.user_labels = None

    def add_points(self, inputs=None, targets=None, time_series=None, summary_stats=None, user_labels=None):
        super(DataSetMET, self).add_points(self, inputs=None, targets=None, time_series=None, summary_stats=None)

        if user_labels is not None:
			if self.user_labels is not None:
				self.user_labels = np.concatenate((self.user_labels, user_labels), axis=0)
			else:
				self.user_labels = user_labels

class StochMET():

    def __init__(self, simulator=None, sampling=None, features=None, default_batch_size=10):
        assert simulator is not None, "simulator not defined" 
        assert sampling is not None, "sampling not defined"
        self.simulator = simulator
        self.sampling = sampling #TODO: check InitialDesignBase
        if features is None:
            self.features = MinimalFCParameters()
        else:
            self.features = features #TODO: check supported format
        
        self.batch_size = default_batch_size
        self.data = DataSetMET()
        self.summaries = SummariesTSFRESH()

    def compute(self, n_points=None):
        if n_points is None:
            n_points = self.batch_size
        
        params = self.sampling.generate(n_points)
        trajectories = simulator(params)

        features = summaries.compute(trajectories)


        

