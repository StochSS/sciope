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
"""
Model Exploration
"""

# Imports 
from sciope.utilities.summarystats.summary_base import SummaryBase
from sciope.features.feature_extraction import generate_tsfresh_features
from sciope.designs.initial_design_base import InitialDesignBase
from sciope.designs.latin_hypercube_sampling import LatinHypercube
from sciope.designs.factorial_design import FactorialDesign
from sciope.utilities.summarystats.summary_base import SummaryBase
from sciope.sampling.sampling_base import SamplingBase
from sciope.utilities.priors.prior_base import PriorBase
from sciope.visualize.interactive_scatter import interative_scatter
from tsfresh.feature_extraction import MinimalFCParameters
from sciope.data.dataset import DataSet
from sciope.core import core
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from dask import persist, delayed, compute
from dask.distributed import as_completed, futures_of
import numpy as np
import umap
from itertools import combinations



def _do_tsne(data, nr_components=2, init='random', plex=30,
             n_iter=1000, lr=200, rs=None):
    """Uses sklearn TSNE non-linear dimension reduction method
    
    Parameters
    ----------
    data : ndarray
    nr_components : int, optional
        desired dimension, by default 2
    init : str, optional
        see sklearn.manifold.t_sne documentation, by default 'random'
    plex : int, optional
        perplexity see sklearn.manifold.t_sne documentation, by default 30
    n_iter : int, optional
        see sklearn.manifold.t_sne documentation, by default 1000
    lr : int, optional
        learning_rate, see sklearn.manifold.t_sne documentation, by default 200
    rs : int, optional
        random seed, by default None
    
    Returns
    -------
    tuple, (transformed data, model)
    
    """

    tsne = TSNE(n_components=nr_components, init=init,
                      perplexity=plex, random_state=rs, n_iter=n_iter, learning_rate=lr)

    return tsne.fit_transform(data), tsne


def _do_pca(data, nr_components=2, rs=None):
    """Uses sklearn PCA dimension reduction method
    
    Parameters
    ----------
    data : ndarray
    nr_components : int, optional
        desired dimension, by default 2
    rs : int, optional
        random seed, by default None
    
    Returns
    -------
    tuple, (transformed data, model)
    """
    pca = PCA(n_components=nr_components, random_state=rs)
    return pca.fit_transform(data), pca


def _do_kpca(data, nr_components=2, kernel='rbf', gamma=0.01,
             degree=3):
    """Uses sklearn kernel-PCA dimension reduction method
    
    Parameters
    ----------
    data : ndarray
    nr_components : int, optional
        desired dimension, by default 2
    kernel : str, optional
        desired kernel see sklearn documentation, by default 'rbf'
    gamma : float, optional
        see sklearn documentation, by default 0.01
    degree : int, optional
        see sklearn documentation, by default 3

    Returns
    -------
    tuple, (transformed data, model)
    """

    kpca = KernelPCA(n_components=nr_components, kernel=kernel, gamma=gamma,
                     degree=degree)
    return kpca.fit_transform(data), kpca


def _do_umap(data, nr_components=2, nr_neighbors=10, min_dist=0.1):
    """Uses UMAP non-linear dimension reduction method
    
    Parameters
    ----------
    data : ndarray
    nr_components : int, optional
        desired dimension, by default 2
    nr_neighbors : int, optional
        see umap documentation, by default 10
    min_dist : float, optional
        see umap documentation, by default 0.1
    
    Returns
    -------
    tuple, (transformed data, model)
    """
    rd = umap.UMAP(n_components=nr_components, n_neighbors=nr_neighbors,
                   min_dist=min_dist)
    return rd.fit_transform(data), rd


def _validate_dr_method(method):
    """validate supported dimension reduction methods,
    called in _do_dimension_reduction
    
    Parameters
    ----------
    method : string
    
    Raises
    ------
    ValueError
        if method is not supported
    """
    allowed_methods = ["umap", "t_sne", "pca", "kpca"]
    if method not in allowed_methods:
        raise ValueError("Supported dimension reduction methods are: {0} "
                         " got dr_method={1}".format(allowed_methods,
                                                     method))


def _do_dimension_reduction(data, method, kwargs={}):
    """Perform dimension reduction method
    
    Parameters
    ----------
    data : ndarray
    method : string
        dimension reduction method, supported ["umap", "t_sne", "pca", "kpca"]
    kwargs : dict, optional
        optional parameters to method, by default {}

    Raises
    ------
    ValueError
        if method is not supported
    
    Returns
    -------
    tuple, (transformed data, model)
    
    """
    _validate_dr_method(method)
    if method == 'umap':
        return _do_umap(data, **kwargs)
    if method == 't_sne':
        return _do_tsne(data, **kwargs)
    if method == 'pca':
        return _do_pca(data, **kwargs)
    else:
        return _do_kpca(data, **kwargs)


class DataSetMET(DataSet):
    """ 
    DataSet class. Container for keeping MET results. 
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

    sim : function which takes a parameter point (generated by "sampler",
                see below) and returns simulation results in the form of
                trajectories (time series) with shape (n_timepoints, n_species)
    sampler :   Instance of LatinHypercube, FactorialDesign or PriorBase
                TODO: add support for all "InitialDesignBase", "PriorBase", "SamplingBase"
    summarystats :  Instance of SummaryBase

    default_batch_size : int, sets the default batch size (number of points computed) of the
                         parameter sweeps. Default is 10.
    default_chunk_size : int, sets the default chunk size

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

    def __init__(self, sim, sampler, summarystats, default_batch_size=100, default_chunk_size=1):

        assert callable(sim), "simulator must be a callable function"

        allowed_sampler = ["LatinHypercube", "FactorialDesign", "PriorBase"]
        assert isinstance(sampler,
                          (LatinHypercube, FactorialDesign, PriorBase)), "sampling must be an instance of: {0}".format(
            allowed_sampler)
        
        allowed_stats = ["SummaryBase"]
        assert isinstance(summarystats,
                          SummaryBase), "summarystats must be an instance of: {0}".format(
            allowed_stats)

        self.simulator = sim
        self.sampling = sampler
        self.batch_size = default_batch_size
        self.chunk_size = default_chunk_size
        self.data = DataSetMET()
        self.summaries = summarystats

    def compute(self, n_points=None, chunk_size=None, predictor=None):
        """
        Computes a batch of the parameter sweep.

        Parameters
        ----------
        n_points : int, optional. The batch size of the sweep. Defaults to default_batch_size.
        chunk_size : int, sets the chunk size
        predictor : function, optional. Use a model predictor based on the features as input as the 
                    final step of the workflow. The predictor function must take an array with the
                    same length as the joined feature output. 
                    TODO: currently only supports joined features    

        """
        cluster_mode = core._cluster_mode()
        if n_points is None:
            n_points = self.batch_size
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        graph_dict = core.get_graph_chunked(self.sampling.draw, 
                                            self.simulator,
                                            self.summaries.compute, 
                                            batch_size=n_points,
                                            chunk_size=chunk_size)
        pred = []
        if predictor is not None:
            if callable(predictor):
                pred = core.get_prediction(predictor, graph_dict["summarystats"])
            else:
                raise ValueError("The predictor must be a callable function")
            # persist at workers, will run in background
            if cluster_mode:
                params_res, processed_res, result_res, pred_res = persist(graph_dict["parameters"], 
                                                                          graph_dict["trajectories"], 
                                                                          graph_dict["summarystats"],
                                                                          pred)
                # convert to futures
                futures = core.get_futures(result_res)
                f_pred = core.get_futures(pred_res)
                f_params = core.get_futures(params_res)
                f_ts = core.get_futures(processed_res)

                # keep track of indices...
                f_dict = {f.key: idx for idx, f in enumerate(f_pred)}
                # ..as we collect result on a "as completed" basis
                for f, pred in as_completed(f_pred, with_results=True):
                    idx = f_dict[f.key]
                    # get the parameter point
                    params = f_params[idx].result()
                    # get the trajatories
                    trajs = f_ts[idx].result()
                    #get summary stats
                    stats = futures[idx].result()
                    # add to data collection
                    param = np.asarray(params)
                    traj = np.asarray(trajs)
                    stats = np.asarray(stats)
                    pred = np.asarray(pred)
                    self.data.add_points(inputs=param, time_series=traj,
                                        summary_stats=stats, user_labels=np.ones(len(stats))*-1,
                                         targets=pred)
            else:
                params_res, processed_res, result_res, pred_res = compute(graph_dict["parameters"], 
                                                                          graph_dict["trajectories"], 
                                                                          graph_dict["summarystats"],
                                                                          pred)
                for e, pred in enumerate(pred_res):
                    param = np.asarray(params_res[e])
                    ts = np.asarray(processed_res[e])
                    stats = np.asarray(result_res[e])
                    pred = np.asarray(pred)
                    self.data.add_points(inputs=param, time_series=ts,
                                         summary_stats=stats, user_labels=np.ones(len(pred))*-1,
                                         targets=pred)


        else:
            # TODO: avoid redundancy...
            if cluster_mode:
                params_res, processed_res, result_res = persist(graph_dict["parameters"], 
                                                                graph_dict["trajectories"], 
                                                                graph_dict["summarystats"])

                # convert to futures
                futures = core.get_futures(result_res)
                f_params = core.get_futures(params_res)
                f_ts = core.get_futures(processed_res)

                # keep track of indices...
                f_dict = {f.key: idx for idx, f in enumerate(futures)}
                # ..as we collect result on a "as completed" basis
                for f, res in as_completed(futures, with_results=True):
                    idx = f_dict[f.key]
                    # get the parameter point
                    params = f_params[idx].result()
                    # get the trajatories
                    trajs = f_ts[idx].result()
                    # add to data collection
                    param = np.asarray(params)
                    traj = np.asarray(trajs)
                    res = np.asarray(res)
                    self.data.add_points(inputs=param, time_series=traj,
                                        summary_stats=res, user_labels=np.ones(len(res))*-1)
            else:
                params_res, processed_res, result_res = compute(graph_dict["parameters"], 
                                                                          graph_dict["trajectories"], 
                                                                          graph_dict["summarystats"])
                for e, res in enumerate(result_res):
                    param = np.asarray(params_res[e])
                    ts = np.asarray(processed_res[e])
                    res = np.asarray(res)
                    self.data.add_points(inputs=param, time_series=ts,
                                         summary_stats=res, user_labels=np.ones(len(res))*-1)

            

    def explore(self, dr_method='umap', scaling=None, kwargs={}):
        """
        Visualize the results from the total parameter sweep.

        Parameters
        ----------

        dr_method : String, optional. Dimension reduction method to use. Supported methods
                     are 'umap', 't_sne', 'pca' and 'kpca' (kernel pca). Default is 'umap'

        scaling : class, optional. Class containing method 'fit_transform' (see sklearn).

        kwargs : TODO: optional parameters to dimension reduction method 
        
        """
        data = self.data.s.reshape(self.data.s.shape[0],self.data.s.shape[-1])
        if scaling is not None:
            assert hasattr(scaling, 'fit_transform'), "%r.fit_transform does not exist" % scaling
            data = scaling.fit_transform(data)
       
        data.astype(np.float32)
        data, model = _do_dimension_reduction(data, dr_method, **kwargs)
        self.dr_model = model
        interative_scatter(data,
                           self.data)
