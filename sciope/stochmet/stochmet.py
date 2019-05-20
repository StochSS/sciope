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

from sciope.utilities.summarystats.summary_base import SummaryBase
from sciope.features.feature_extraction import generate_tsfresh_features
from sciope.visualize.interactive_scatter import interative_scatter
from tsfresh.feature_extraction import MinimalFCParameters
from sciope.data.dataset import DataSet
from sklearn.manifold import t_sne
from sklearn.decomposition import PCA, KernelPCA
from dask import persist, delayed
import numpy as np
import umap
from itertools import combinations




def _do_tsne(data, nr_components = 2, init = 'random', plex = 30,
        n_iter = 1000, lr = 200, rs= None):
        """[summary]
        
        Parameters
        ----------
        data : [type]
            [description]
        nr_components : int, optional
            [description], by default 2
        init : str, optional
            [description], by default 'random'
        plex : int, optional
            [description], by default 30
        n_iter : int, optional
            [description], by default 1000
        lr : int, optional
            [description], by default 200
        rs : [type], optional
            [description], by default None
        """

    tsne = t_sne.TSNE(n_components=nr_components, init=init,
            perplexity=plex, random_state=rs, n_iter=n_iter, learning_rate=lr)

    return tsne.fit_transform(data), tsne

def _do_pca(data, nr_components = 2, rs = None):
    """[summary]
    
    Parameters
    ----------
    data : [type]
        [description]
    nr_components : int, optional
        [description], by default 2
    rs : [type], optional
        [description], by default None
    """
    pca = PCA(n_components=nr_components , random_state = rs)
    return pca.fit_transform(data), pca

def _do_kpca(data , nr_components  = 2 , kernel  = 'rbf', gamma = 0.01,
        degree = 3):
    """[summary]
    
    Parameters
    ----------
    data : [type]
        [description]
    nr_components : int, optional
        [description], by default 2
    kernel : str, optional
        [description], by default 'rbf'
    gamma : float, optional
        [description], by default 0.01
    degree : int, optional
        [description], by default 3
    """

    kpca = KernelPCA(n_components = nr_components, kernel = kernel, gamma = gamma,
            degree = degree)
    return kpca.fit_transform(data), kpca

def _do_umap(data, nr_components = 2, nr_neighbors = 10, min_dist = 0.1):
    """[summary]
    
    Parameters
    ----------
    data : [type]
        [description]
    nr_components : int, optional
        [description], by default 2
    nr_neighbors : int, optional
        [description], by default 10
    min_dist : float, optional
        [description], by default 0.1
    """
    rd = umap.UMAP(n_components = nr_components, n_neighbors = nr_neighbors,
            min_dist = min_dist)
    return rd.fit_transform(data), rd

def _validate_dr_method(method):
    """[summary]
    
    Parameters
    ----------
    method : [type]
        [description]
    
    Raises
    ------
    ValueError
        [description]
    """
        allowed_methods = ["umap", "t_sne", "pca", "kpca"]
        if method not in allowed_methods:
            raise ValueError("Implemented dimension reduction methods are: {0} "
                             " got dr_method={1}".format(allowed_methods,
                                                        method))

def _do_dimension_reduction(X, method, kwargs={}):
    """[summary]
    
    Parameters
    ----------
    X : [type]
        [description]
    method : [type]
        [description]
    kwargs : dict, optional
        [description], by default {}
    """
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


    def distribute(self, point):
        """
        Computes features for one point (time series).
        
        Parameters
        ----------
        point : ndarray
            trajectory of shape n_timepoints x 1

        Returns
        -------
        list
            list of generated features 
        """
        return list(generate_tsfresh_features(data=[point], features=self.features)[0])
        

    def correlation(self, x, y):
        """
        Computes the Pearson correlation coefficient between two trajectories
        
        Parameters
        ---------

        x : ndarray 
            Trajectory of shape n_timepoints x 1 

        y: ndarray 
            Trajectory of shape n_timepoints x 1 

        Returns
        list
            list of generated feature
        """
        return [np.corrcoef(x,y)[0,1]]


class DataSetMET(DataSet):
    """ 
    DataSet class. Container for keeping MET results in memory. 
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

    Parameters
    ----------

    simulator : function which takes a parameter point (generated by "sampler",
                see below) and returns simulation results in the form of
                trajectories (time series) with shape (n_timepoints, n_species)
    sampler :   function with parameter "n_points" as int which returns a ndarray of n_points
                parameter points
                TODO: support both user-defined functions and built-in options as string
    features :  Dictionary, containing tsfresh features to be computed. Sets the default_fc_parameters used 
                in tsfresh. For example: 
                fc_parameters = {
                                "length": None,
                                "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}]
                                }
                See tsfresh documentation for supported features.
                Default is {'sum_values': None,
                            'median': None,
                            'mean': None,
                            'standard_deviation': None,
                            'variance': None,
                            'maximum': None,
                            'minimum': None}

    default_batch_size : int, sets the default batch size of the parameter sweeps. Default is 10.

    Attributes
    ----------
    data : Local data container stored in local memory, which holds the results from each batch.
           Need to call class methods "explore" or "_collect_persisted" to collect data from persisted
           storage after "compute" has been called.

    features : tsfresh features currently used. Obs! Changing this attribute after a batch with different
               features will raise IndexError during data concatenation. Make sure to save your data or
               manipulate the "s" attribute in data so that it is coherent with the newly set features.
               TODO: future versions will handle this problem automatically 
      

    """

    def __init__(self, simulator=None, sampler=None, features=None, default_batch_size=10):
        assert callable(simulator), "simulator must be a callable function" 
        assert hasattr(sampler, 'generate'), "sampling class instance must have a callable function 'generate'"
        self.simulator = simulator
        self.sampling = sampler 
        self.batch_size = default_batch_size
        self.data = DataSetMET()
        self.summaries = SummariesTSFRESH()
        if features is None:
            self.features = MinimalFCParameters()
            self.features.pop('length')
        else:
            self.features = features #TODO: check supported format
        self.summaries.features = self.features

    def compute(self, n_species, n_points=None, join_features=True, predictor=None):
        """
        Computes a batch of the parameter sweep.

        Parameters
        ----------

        n_species : int or array-like. If int, will compute features of all 
                    species with index in the list range(n_species) based on the output from
                    simulator with shape (n_timepoints, n_species). If array-like, explicitly
                    set which indices to compute features on.
                    

        n_points : int, optional. The batch size of the sweep. Defaults to default_batch_size.
        join_features : boolean. Wheather features of each species should be joined into a single
                        array. Default is True.
        predictor : function, optional. Use a model predictor based on the features as input as the 
                    final step of the workflow. The predictor function must take an array with the
                    same length as the joined feature output. 
                    TODO: currently only supports joined features    

        """
        if n_points is None:
            n_points = self.default_batch_size
        #da_params = da.asarray(self.sampling.generate(n_points))
        sampler = delayed(self.sampling.generate) # according to best practice instead of passing a
                                                  # dask collection to a delayed object
        params = [sampler(1)[0] for x in range(n_points)]

        simulator = delayed(self.simulator)
        
        features = delayed(self.summaries.distribute)
            
        processed = [simulator(g) for g in params]    
        
        if type(n_species) is int:
            n_species = range(n_species)
        all_features = []
        
        for p in processed:
            for s in n_species: #try catch EventFired
                traj = p[:,s] #get_item
                all_features.append(features(traj))

            if hasattr(self.summaries, 'correlation'):
                correlation = delayed(self.summaries.correlation)
                for s in combinations(n_species, 2):
                    x = p[:, s[0]] #get_item
                    y = p[:, s[1]] #get_item
                    all_features.append(correlation(x,y))
        
        result = []
        if join_features:
            window_len = len(n_species) + len(list(combinations(n_species, 2)))
            
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
    

    def explore(self, dr_method='umap', scaling=None, from_distributed=True, filter_func=None, kwargs={}):
        """
        Visualize the results from the total parameter sweep.

        Parameters
        ----------

        dr_method : String, optional. Dimension reduction method to use. Supported methods
                     are 'umap', 't_sne', 'pca' and 'kpca' (kernel pca). Default is 'umap'

        scaling : class, optional. Class containing method 'fit_transform' (see sklearn).

        from_distributed : boolean. Collect data from persited storage. Obs! If all data 
                           has already been collected and from_distributed = True an error 
                           will be raised complaining that no futures exists, set 
                           from_distributed to False in this case. 
                           TODO: future versions will simplify this

        filter_func : function, optional. A function that takes the output from "predictor" 
                      (if used in "compute") and filter persited data according to some 
                      criteria (e.g entropy for active learning or predicted class). 
                      The function should return True if the criteria is statisfied for one
                      individual point, and False otherwise.
        kwargs : TODO: parameters for dr_method 
        
        """
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
        

    def _collect_persisted(self, filter_func=None):
        """
        Collects data from persited storage and store it in StochMET.data
        
        Parameters
        ----------
        filter_func : function, optional. A function that takes the output from "predictor" 
                      (if used in "compute") and filter persited data according to some 
                      criteria (e.g entropy for active learning or predicted class). 
                      The function should return True if the criteria is statisfied for one
                      individual point, and False otherwise. By default None.
        """
        assert hasattr(self, 'futures'), "There is no futures (data) to be collected"
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



        

