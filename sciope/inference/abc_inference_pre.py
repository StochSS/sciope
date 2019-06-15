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
from sciope.utilities.housekeeping import sciope_logger as ml
from sciope.utilities.housekeeping import sciope_profiler
from sciope.data.dataset import DataSet
from toolz import partition_all
import multiprocessing as mp  # remove dependency
import numpy as np
import dask

# The following variable stores n normalized distance values after n summary statistics have been calculated
normalized_distances = None


# Class definition: multiprocessing ABC process
class ABCProcess(mp.Process):
    """
    The process class used to distribute the sampling process
    """

    def run(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


# Class definition: ABC rejection sampling
class ABC():
    """
    Approximate Bayesian Computation Rejection Sampler
    * InferenceBase.infer()
    """

    def __init__(self, data, param, time_series, epsilon=0.1, summaries_function=bs.Burstiness(),
                 distance_function=euc.EuclideanDistance(), use_logger=False):
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
        self.name = 'ABC with pre generated data'
        self.epsilon = epsilon
        self.summaries_function = summaries_function
        self.distance_function = distance_function
        self.fixed_mean = []
        self.use_logger = use_logger

        self.time_series = time_series
        self.param = param
        self.data = data

        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Approximate Bayesian Computation initialized")

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

    def get_dask_graph(self, batch_size):
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
        # trial_param = [self.prior_function.draw() for x in range(batch_size)]



        # Perform the trial
        # sim_result = [self.sim(param) for param in trial_param]

        # Draw prior and timeseries from dataset
        trial_param, sim_result = self.dataset

        # Get the statistic(s)
        sim_stats = [self.summaries_function.compute([sim]) for sim in sim_result]

        # Calculate the distance between the dataset and the simulated result
        sim_dist = [self.distance_function.compute(self.fixed_mean, stats) for stats in sim_stats]

        return {"parameters": trial_param, "trajectories": sim_result, "summarystats": sim_stats, "distances": sim_dist}

    # @sciope_profiler.profile
    def rejection_sampling(self, num_samples, batch_size, chunk_size):
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
        graph_dict = self.get_dask_graph(batch_size)

        # do rejection sampling
        while accepted_count < num_samples:

            res_param, res_dist = dask.compute(graph_dict["parameters"], graph_dict["distances"])

            # Normalize distances between [0,1]
            sim_dist_scaled = np.asarray([self.scale_distance(dist) for dist in res_dist])

            # Take the norm to combine the distances, if more than one summary is used
            if sim_dist_scaled.shape[1] > 1:
                combined_distance = [dask.delayed(np.linalg.norm)(scaled.reshape(1, scaled.size), axis=1)
                                     for scaled in sim_dist_scaled]
                result, = dask.compute(combined_distance)
            else:
                result = sim_dist_scaled.ravel()

            # Accept/Reject
            for e, res in enumerate(result):
                if self.use_logger:
                    self.logger.debug("ABC Rejection Sampling: trial parameter(s) = {}".format(res_param[e]))
                    self.logger.debug("ABC Rejection Sampling: trial distance(s) = {}".format(res_dist[e]))
                if res <= self.epsilon:
                    accepted_samples.append(res_param[e])
                    distances.append(res_dist[e])
                    accepted_count += 1
                    if self.use_logger:
                        self.logger.info("ABC Rejection Sampling: accepted a new sample, "
                                         "total accepted samples = {0}".format(accepted_count))

            trial_count += batch_size

        self.results = {'accepted_samples': accepted_samples, 'distances': distances, 'accepted_count': accepted_count,
                        'trial_count': trial_count, 'inferred_parameters': np.mean(accepted_samples, axis=0)}
        return self.results
    def infer2(self,num_samples):
        print("infer2")

    def infer(self, num_samples):
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
        print("welcome to inf")
        print("time series shape: ", self.time_series.shape)


        data_s = [self.summaries_function.compute(x) for x in self.time_series]

        data_s = dask.compute(data_s)

        print("data_s shape", np.array(data_s).shape)
        max_s = np.max(data_s, axis=1)
        min_s = np.min(data_s, axis=1)
        print("max s: ", max_s, ", min s: ", min_s)
        #normalized
        data_s = (data_s-min_s)/(max_s-min_s)

        obs_data_s = (self.summaries_function.compute(self.data)-min_s)/(max_s-min_s)

        distances = self.distance_function.compute(data_s, obs_data_s).compute()

        print("distances shape: ", distances.shape)



