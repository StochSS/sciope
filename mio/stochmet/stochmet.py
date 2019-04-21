# Copyright 2017 Prashant Singh, Fredrik Wrede and Andreas Hellander
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mio.utilities.summarystats.summary_base import SummaryBase
from mio.features.feature_extraction import generate_tsfresh_features
from mio.visualize.interactive_scatter import interative_scatter
from tsfresh.feature_extraction import MinimalFCParameters
from mio.data.dataset import DataSet
from sklearn.manifold import t_sne
from dask.distributed import Client
from sklearn.decomposition import PCA, KernelPCA
from toolz import partition_all
from dask import persist, delayed
import numpy as np
import umap
from itertools import combinations




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
    
class EventFired(Exception):
    """ 
    Exception class to handle events in solvers
    
    """
    pass

class SummariesTSFRESH(SummaryBase):
    """
    Class for computing features/statistics on time series data.
    An ensemble of different statistics from TSFRESH are supported.
    """

    def __init__(self):
        self.name = 'SummariesTSFRESH'
        self.features = MinimalFCParameters()
        self.features.pop('length')
        super(SummariesTSFRESH, self).__init__(self.name)

    def compute(self, data, features=MinimalFCParameters(), dask_client=None, chunk_size=1):
        """ 
        Generate features in local mode based on batches of parameter points

        Parameters
        ---------

        data : numpy.ndarray of shape n_points x n_timepoints x n_species


        """
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

    def distribute(self, point):
        """
        Computes features for one point (time series).

        Paramters
        ---------

        point : numpy.ndarray of shape n_timepoints x 1 
        
        Returns
        params : list of features

        """
        #f = MinimalFCParameters()
        #f.pop('length')
        return list(generate_tsfresh_features(data=[point], features=self.features)[0])
        

    def correlation(self, x, y):
        """
        Computes the Pearson correlation coefficient between two time series
        
        Paramters
        ---------

        x : numpy.ndarray of shape n_timepoints x 1 

        y: numpy.ndarray of shape n_timepoints x 1 

        Returns
        params : list with feature
        """
        return [np.corrcoef(x,y)[0,1]]


class DataSetMET(DataSet):
    """ 
    DataSet class
    """

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
    """ 
    Stochastic Model Exploration Toolkit (StochMET)

    """

    def __init__(self, simulator=None, sampling=None, features=None, default_batch_size=10):
        assert callable(simulator), "simulator must be a callable function" 
        assert hasattr(sampling, 'generate'), "sampling class instance must have a callable function 'generate'"
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
    
    def compute_futures(self, n_points=10, dask_client=None, chunk_size=10, only_features=False):
        
        # Draw parameter points 
        params = self.sampling.generate(n_points)
        params_chunks = partition_all(chunk_size, params)
        
        f = dask_client.map(self.simulator, params_chunks)
        if only_features:
            f_features = generate_tsfresh_features(f, features=self.features, 
                                dask_client=dask_client, chunk_size=chunk_size)

        else:
            res_ts = dask_client.gather(f)
            f_features = generate_tsfresh_features(res_ts, features=self.features, 
                                dask_client=dask_client, chunk_size=chunk_size)

        res_features = dask_client.gather(f_features)
        if only_features:
            return list(res_features)
        else:
            return list(res_features), list(res_ts)

    def compute_delay(self, n_points, n_species, join_features=True, predictor=None):

        #da_params = da.asarray(self.sampling.generate(n_points))
        sampler = delayed(self.sampling.generate) # according to best practice instead of passing a
                                                  # dask collection to a delayed object
        params = [sampler(1)[0] for x in range(n_points)]

        simulator = delayed(self.simulator)
        
        features = delayed(self.summaries.distribute)
            
        processed = [simulator(g) for g in params]    

        species_lst = range(n_species)
        all_features = []
        
        for p in processed:
            for s in species_lst: #try catch EventFired
                traj = p[:,s] #get_item
                all_features.append(features(traj))

            if hasattr(self.summaries, 'correlation'):
                correlation = delayed(self.summaries.correlation)
                for s in combinations(species_lst, 2):
                    x = p[:, s[0]] #get_item
                    y = p[:, s[1]] #get_item
                    all_features.append(correlation(x,y))
        
        result = []
        if join_features:
            window_len = n_species + len(list(combinations(species_lst, 2)))
            
            @delayed
            def join(lst):
                joined = lst[0]
                for item in lst[1:]:
                    joined += item
                return joined
        
            for j in range(0, len(all_features), window_len):
                result.append(join(all_features[j:j+window_len]))
        else:
            result = all_features
        
        pred = []
        if predictor is not None:
            if callable(predictor):
                predictor = delayed(predictor)
                pred = [predictor(x) for x in result]
            else:
                raise ValueError("The predictor must be a callable function")
            #persist at workers, will run in background
            params_res, processed_res, result_res, pred_res = persist(params, processed, result, pred)
            #keep on workers until needed for local processing
            self.futures = {'parameters': params_res, 'ts': processed_res, 'features': result_res,
                            'prediction': pred_res}
        else:
            #TODO: avoid redundancy...
            params_res, processed_res, result_res = persist(params, processed, result)
            self.futures = {'parameters': params_res, 'ts': processed_res, 'features': result_res} 
    
    def explore(self, dr_method='umap', scaling=None, from_distributed=False, filter_func=None, kwargs={}):
        if from_distributed:
            # collecting data from distributed RAM. TODO: "explore" should read only neccesary data
            self._collect_persisted(filter_func)
            del self.futures

        if scaling is not None:
            assert hasattr(scaling, 'fit_transform'), "%r.fit_transform does not exist" % scaling
            data = scaling.fit_transform(self.data.s)
        else:
            data = self.data.s
        
        data.astype(np.float32)
        data, model = _do_dimension_reduction(data, dr_method, **kwargs)
        self.dr_model = model
        interative_scatter(data, self.data) #TODO: interactive_scatter now treat DataSet.y as labels, change to DataSet.user_labels
        

    def _collect_persisted(self, filter_func):
        assert hasattr(self, 'futures'), "There is no data on the cluster"
        use_filter = False
        if filter_func is not None:
            if callable(filter_func):
                use_filter = True
            else:
                raise ValueError("The filter must be a callable function returning"
                                "True of False")
        for e, i in enumerate(self.futures['ts']):    
            try:
                ts = np.array([i.compute()])
                if 'prediction' in self.futures.keys():
                    pred = self.futures['prediction'][e].compute()
                    if use_filter:
                        if filter_func(pred):
                            self.data.add_points(targets=np.array([pred]))
                        else:
                            continue

                param = np.array([self.futures['parameters'][e].compute()])
                feature = np.array([self.futures['features'][e].compute()])
               
                self.data.add_points(inputs=param, time_series=ts, 
                            summary_stats=feature, user_labels=np.array([-1]))
            except EventFired:
                continue



        

