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
Approximate Bayesian Computation
"""

# Imports
from sciope.inference.inference_base import InferenceBase
from sciope.utilities.distancefunctions import euclidean as euc
from sciope.utilities.summarystats import burstiness as bs
# from sciope.utilities.housekeeping import sciope_logger as ml
# from sciope.utilities.housekeeping import sciope_profiler
from sciope.data.dataset import DataSet
from toolz import partition_all
import multiprocessing as mp  # remove dependency
import numpy as np
import dask

# The following variable stores n normalized distance values after n summary statistics have been calculated
normalized_distances = None


# Set up the logger
# logger = ml.SciopeLogger().get_logger()


# Class definition: multiprocessing ABC process
class ABCProcess(mp.Process):
    """
    The process class used to distribute the sampling process
    """

    def run(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


# Class definition: ABC rejection sampling
class ABC(InferenceBase):
    """
    Approximate Bayesian Computation Rejection Sampler

    * InferenceBase.infer()
    """

    def __init__(self, data, sim, prior_function, epsilon=0.1, parallel_mode=False, summaries_function=bs.Burstiness(),
                 distance_function=euc.EuclideanDistance()):
        """
        ABC class for rejection sampling
        
        Parameters
        ----------
        data : [type]
            [description]
        sim : [type]
            [description]
        prior_function : [type]
            [description]
        epsilon : float, optional
            [description], by default 0.1
        parallel_mode : bool, optional
            [description], by default False
        summaries_function : [type], optional
            [description], by default bs.Burstiness()
        distance_function : [type], optional
            [description], by default euc.EuclideanDistance()
        """
        self.name = 'ABC'
        self.epsilon = epsilon
        self.summaries_function = summaries_function
        self.prior_function = prior_function
        self.distance_function = distance_function
        self.parallel_mode = parallel_mode
        self.historical_distances = []
        self.fixed_mean = []
        super(ABC, self).__init__(self.name, data, sim)
        self.sim = dask.delayed(sim)
        # logger.info("Approximate Bayesian Computation initialized")

    def scale_distance(self, dist):
        """
         Performs scaling in [0,1] of a given distance vector/value with respect to historical distances
        
        Parameters
        ----------
        dist : ndarray, float
            distance
        
        Returns
        -------
        ndarray
            scaled distance
        """

        dist = np.asarray(dist)
        global normalized_distances
        self.historical_distances.append(dist.ravel())
        all_distances = np.array(self.historical_distances)
        divisor = np.asarray(np.nanmax(all_distances, axis=0))
        normalized_distances = all_distances
        for j in range(0, len(divisor), 1):
            if divisor[j] > 0:
                normalized_distances[:, j] = normalized_distances[:, j] / divisor[j]

        return normalized_distances[-1, :]

    def compute_fixed_mean(self, chunk_size):
        """
        Computes the mean over summary statistics on fixed data
        
        Parameters
        ----------
        chunk_size : int
            the partition size when splitting the fixed data. For avoiding many individual tasks
            in dask if the data is large 
        
        Returns
        -------
        ndarray
            scaled distance
        """

        # assumed data is large, make chunks
        data_chunked = partition_all(chunk_size, self.data)

        # compute summary stats on fixed data
        stats = [self.summaries_function.compute(x) for x in data_chunked]

        mean = dask.delayed(np.mean)

        # reducer 1 mean for each batch
        stats_mean = mean(stats, axis=0)

        # reducer 2 mean over batches
        stats_mean = mean(stats_mean, axis=0, keepdims=True).compute()

        self.fixed_mean = np.copy(stats_mean)
        del stats_mean

    def get_dask_graph(self, batch_size, ensemble_size=1):
        """
        Constructs the dask computational graph invloving sampling, simulation, summary statistics
        and distances.
        
        Parameters
        ----------
        batch_size : int
            The number of points being sampled in each batch.
        
        Returns
        -------
        dict
            with keys 'parameters', 'trajectories', 'summarystats' and 'distances'
        """

        # Rejection sampling with batch size = batch_size 

        # Draw from the prior
        trial_param = [self.prior_function.draw() for x in range(batch_size)]*ensemble_size

        # Perform the trial
        sim_result = [self.sim(param) for param in trial_param]

        # Get the statistic(s)
        sim_stats = [self.summaries_function.compute([sim]) for sim in sim_result]

        if ensemble_size > 1:

            def window(iterable, size):
                i = iter(iterable)
                win = []
                for e in range(0, size):
                    win.append(next(i))
                yield win
                for e in i:
                    win = win[1:] + [e]
                    yield win
            
            w = window(sim_stats, ensemble_size)
            stats_final = [dask.delayed(np.mean)(x, axis=0) for x in w]
        else:
            stats_final = sim_stats

        # Calculate the distance between the dataset and the simulated result
        sim_dist = [self.distance_function.compute(self.fixed_mean, stats) for stats in stats_final]

        return {"parameters": trial_param, "trajectories": sim_result, "summarystats": stats_final, "distances": sim_dist}

    # @sciope_profiler.profile
    def rejection_sampling(self, num_samples, batch_size, chunk_size, ensemble_size=1):
        """
        Perform ABC inference according to initialized configuration.

        Parameters
        ----------
        num_samples : int
            The number of required accepted samples
        batch_size : int
            The batch size of samples for performing rejection sampling
        chunk_size : int
            the partition size when splitting the fixed data. For avoiding many individual tasks
            in dask if the data is large.
        
        Returns
        -------
        dict
            Keys
            'accepted_samples: The accepted parameter values', 
            'distances: Accepted distance values', 
            'accepted_count: Number of accepted samples',
            'trial_count: The number of total trials performed in order to converge',
            'inferred_parameters': The mean of accepted parameter samples
        """
        accepted_count = 0
        trial_count = 0
        accepted_samples = []
        distances = []

        # if fixed_mean has not been computed
        if not self.fixed_mean:
            self.compute_fixed_mean(chunk_size)

        # Get dask graph
        graph_dict = self.get_dask_graph(batch_size, ensemble_size)

        # do rejection sampling
        while accepted_count < num_samples:

            res_param, res_dist = dask.compute(graph_dict["parameters"], graph_dict["distances"])

            # Normalize distances between [0,1]
            sim_dist_scaled = np.asarray([self.scale_distance(dist) for dist in res_dist])

            # Take the norm to combine the distances, if more than one summary is used
            if sim_dist_scaled.shape[1] > 1:
                combined_distance = [dask.delayed(np.linalg.norm)(scaled, axis=1) for scaled in sim_dist_scaled]
                result, = dask.compute(combined_distance)
            else:
                result = sim_dist_scaled.ravel()

            # Accept/Reject
            for e, res in enumerate(result):
                if res <= self.epsilon:
                    accepted_samples.append(res_param[e])
                    distances.append(res_dist[e])
                    accepted_count += 1

            trial_count += batch_size

        self.results = {'accepted_samples': accepted_samples, 'distances': distances, 'accepted_count': accepted_count,
                        'trial_count': trial_count, 'inferred_parameters': np.mean(accepted_samples, axis=0)}
        return self.results

    def infer(self, num_samples, batch_size, chunk_size=10, ensemble_size=1):
        """
        Wrapper for rejection sampling. Performs ABC rejection sampling
        
        Parameters
        ----------
        num_samples : int
            The number of required accepted samples
        batch_size : int
            The batch size of samples for performing rejection sampling
        chunk_size : int
            the partition size when splitting the fixed data. For avoiding many individual tasks
            in dask if the data is large. Default 10.
        
        Returns
        -------
        dict
            Keys
            'accepted_samples: The accepted parameter values', 
            'distances: Accepted distance values', 
            'accepted_count: Number of accepted samples',
            'trial_count: The number of total trials performed in order to converge',
            'inferred_parameters': The mean of accepted parameter samples
        """

        return self.rejection_sampling(num_samples, batch_size, chunk_size, ensemble_size)
