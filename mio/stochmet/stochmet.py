from mio.utilities.summarystats.summary_base import SummaryBase
from mio.features.feature_extraction import generate_tsfresh_features
from mio.designs.random_sampling import RandomSampling
from mio.visualize.interactive_scatter import interative_scatter
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.manifold import t_sne
from dask.distributed import Client
from sklearn.decomposition import PCA, KernelPCA
from mio.data.dataset import DataSet
from toolz import partition_all
import numpy as np
import umap




def _do_tsne(data, nr_components = 2, init = 'random', plex = 30,
        n_iter = 1000, lr = 200, rs= None):

    tsne = t_sne.TSNE(n_components=nr_components, init=init,
            perplexity=plex, random_state=rs, n_iter=n_iter, learning_rate=lr)

    return tsne.fit_transform(data), tsne

def _do_pca(data, nr_components = 2, rs = None):
    pca = PCA(n_components=nr_components , random_state = rs)
    return pca.fit_transform(data), pca

def _do_kpca(data , nr_components  = 2 , kernel  = 'rbf', gamma = 0.01,
        degree = 3):

    kpca = KernelPCA(n_components = nr_components, kernel = kernel, gamma = gamma,
            degree = degree)
    return kpca.fit_transform(data), kpca

def _do_umap(data, nr_components = 2, nr_neighbors = 10, min_dist = 0.1):
    rd = umap.UMAP(n_components = nr_components, n_neighbors = nr_neighbors,
            min_dist = min_dist)
    return rd.fit_transform(data), rd

def _validate_dr_method(method):
        allowed_methods = ["umap", "t_sne", "pca", "kpca"]
        if method not in allowed_methods:
            raise ValueError("Implemented dimension reduction methods are: {0} "
                             " got dr_method={1}".format(allowed_methods,
                                                        method))

def _do_dimension_reduction(X, method, kwargs={}):
    _validate_dr_method(method)
    if method == 'umap':
        return _do_umap(X, **kwargs)
    if method == 't_sne':
        return _do_tsne(X, **kwargs)
    if method == 'pca':
        return _do_pca(X, **kwargs)
    else:
        return _do_kpca(X, **kwargs)
    

class SummariesTSFRESH(SummaryBase):
    """
    An ensemble of different statistics from TSFRESH
    """

    def __init__(self):
        self.name = 'SummariesTSFRESH'
        self.features = None
        super(SummariesTSFRESH, self).__init__(self.name)

    def compute(self, data, features=MinimalFCParameters(), dask_client=None, chunk_size=1):
        self.features = features
        num_species = data.shape[2]
        num_points = data.shape[0]
        
        #here we aggregate features from several species into one feature vector
        feature_values = []
        for i in range(num_species):
            feature_values.append(generate_tsfresh_features(data[:,:,i], features, 
                                    dask_client, chunk_size))
        
        # ToDo: Check for NaNs
        return feature_values

class DataSetMET(DataSet):

    def __init__(self):
        name = 'stochmet'
        super(DataSetMET, self).__init__(name)
        self.user_labels = None

    def add_points(self, inputs=None, targets=None, time_series=None, summary_stats=None, user_labels=None):
        super(DataSetMET, self).add_points(inputs, targets, time_series, summary_stats)
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

    def compute(self, n_points=None, kwargs=None):
        if n_points is None:
            n_points = self.batch_size
        
        # Draw parameter points 
        params = self.sampling.generate(n_points)
        
        # Containers for simulation output
        res_trajectories = []
        res_params = []

        # Start simulations
        for p in params:
            try:
                trajectories, param = self.simulator(p, **kwargs)
            except EventFired:
                continue #TODO: generalize
            res_trajectories.append(trajectories)
            res_params.append(param)

        res_trajectories = np.array(res_trajectories)
        res_params = np.array(res_params)
        
        # Compute features of result
        features_values = self.summaries.compute(res_trajectories)

        # Aggregate features from several species
        features_values = np.array(features_values)
        if len(features_values.shape) > 1:
            features_comb = features_values[0]
            for i in range(1, len(features_values)):
                features_comb = np.concatenate((features_comb,features_values[i]), axis=1)

        # Add data to DataSetMET container
        labels = np.ones(len(res_params))*-1 #unlabeled instances as -1 
        self.data.add_points(inputs=res_params, time_series=res_trajectories, 
                                summary_stats=features_comb, user_labels=labels)
    
    def distribute(self, n_points=10, dask_client=None, chunk_size=10, only_features=False):
        
        # Draw parameter points 
        params = self.sampling.generate(n_points)
        params_chunks = partition_all(chunk_size, params)
        
        f = dask_client.map(self.simulator, params_chunks)
        if only_features:
            f_features = generate_tsfresh_features(f, features=self.features, 
                                dask_client=dask_client, chunk_size=chunk_size)

        else:
            print("hej")
            #res_ts = dask_client.gather(f)
            #f_features = generate_tsfresh_features(res_ts, features=self.features, 
                                #dask_client=dask_client, chunk_size=chunk_size)

        #res_features = dask_client.gather(f_features)
        if only_features:
            return list(res_features)
        else:
           #return list(res_features), list(res_ts)
            return f

    
    def explore(self, dr_method='umap', scaling=None, kwargs={}):
        if scaling is not None:
            assert hasattr(scaling, 'fit_transform'), "%r.fit_transform does not exist" % scaling
            data = scaling.fit_transform(self.data.s)
        else:
            data = self.data.s
        
        data, model = _do_dimension_reduction(data, dr_method, **kwargs)
        interative_scatter(data, self.data) #TODO: interactive_scatter now treat DataSet.y as labels, change to DataSet.user_labels
        





        

