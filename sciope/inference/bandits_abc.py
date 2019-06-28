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
Multi-Armed Bandit - Approximate Bayesian Computation
"""

# Imports
from sciope.inference.abc_inference import ABC, get_futures, _cluster_mode
from sciope.utilities.mab import mab_direct as md
from sciope.utilities.distancefunctions import euclidean as euc
from sciope.utilities.summarystats import burstiness as bs
from sciope.utilities.housekeeping import sciope_logger as ml
import numpy as np
import dask
from dask.distributed import futures_of, as_completed, wait

# The following variable stores n normalized distance values after n summary statistics have been calculated
normalized_distances = None


def arm_pull(arm_idx):
    """
    Used by MAB algorithms; Each arm corresponds to a summary statistic and an arm pull is simply selection of one
    (or more) summary statistics in inference. Here that corresponds to simply returning the desired arm.
    :param arm_idx: The index into the vector of arms
    :return: -1 * distance value from distances corresponding to the arm_idx, as reward is to be maximized according to
    MABs but in inference we minimize distance.
    """
    global normalized_distances
    return -1 * normalized_distances[-1, arm_idx]


# Class definition: Bandits-ABC rejection sampling
class BanditsABC(ABC):
    """
    ABC rejection sampling with dynamic multi-armed bandit (MAB) assisted summary statistic selection.

    Properties/variables:
    * data						(observed / fixed data)
    * sim   					(simulator function handle)
    * prior_function			(prior over the simulator parameters)
    * mab_variant               (dynamic summary statistic selection using multi-armed bandits)
    * k                         (desired number of selected summary statistics participating in inference)
    * epsilon 	    			(acceptance tolerance bound)
    * summaries_function    	(summary statistics calculation function)
    * distance_function         (function calculating deviation between simulated statistics and observed statistics)
    * use_logger    			(whether logging is enabled or disabled)


    Methods:
    * infer 					(perform parameter inference)
    * rejection_sampling        (parameter inference using rejection sampling)
    """

    def __init__(self, data, sim, prior_function, mab_variant=md.MABDirect(arm_pull), k=1, epsilon=0.1,
                 summaries_function=bs.Burstiness(), distance_function=euc.EuclideanDistance(), use_logger=False):
        """
        BanditsABC class for rejection sampling

        Parameters
        ----------
        data : nd-array
            the observed / fixed dataset
        sim : nd-array
            the simulated dataset or simulator function
        prior_function : sciope.utilities.priors object
            the prior function generating candidate samples
        mab_variant : sciope.utilities.mab object, optional; by default MABDirect
            summary statistic selection using the chosen multi-armed bandit algorithm
        k : integer
            number of desired summary statistics participating in inference
        epsilon : float, optional
            tolerance bound, by default 0.1
        summaries_function : sciope.utilities.summarystats object, optional
            function calculating summary stats over simulated results; by default bs.Burstiness()
        distance_function : sciope.utilities.distancefunctions object, optional
            distance function operating over summary statistics - calculates deviation between observed and simulated
            data; by default euc.EuclideanDistance()
        use_logger : bool
            enable/disable logging
        """
        super().__init__(data, sim, prior_function, epsilon, summaries_function, distance_function, use_logger)
        self.name = 'BanditsABC'
        self.mab_variant = mab_variant
        self.k = k
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Multi-Armed Bandits Approximate Bayesian Computation initialized")

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

    # @sciope_profiler.profile
    def rejection_sampling(self, num_samples, batch_size, chunk_size, ensemble_size, normalize):
        """
        * overrides rejection_sampling of ABC class *
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
        ensemble_size : int
            In case we have an ensemble of responses
        normalize : bool
            Whether summary statistics should be normalized and epsilon be interpreted as a percentage

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
        cluster_mode = _cluster_mode()

        # do rejection sampling
        while accepted_count < num_samples:

            sim_dist_scaled = []
            params = []
            dists = []

            if cluster_mode:
                if self.use_logger:
                    self.logger.info("running in cluster mode")
                res_param, res_dist = dask.persist(graph_dict["parameters"], graph_dict["distances"])

                futures_dist = get_futures(res_dist)
                futures_params = get_futures(res_param)

                keep_idx = {f.key: idx for idx, f in enumerate(futures_dist)}

                for f, dist in as_completed(futures_dist, with_results=True):
                    dists.append(dist)
                    if normalize:
                        # Normalize distances between [0,1]
                        sim_dist_scaled.append(self.scale_distance(dist))
                    idx = keep_idx[f.key]
                    params.append(futures_params[idx].result())

                del futures_dist, futures_params, res_param, res_dist

            else:
                if self.use_logger:
                    self.logger.info("running in parallel mode")
                params, dists = dask.compute(graph_dict["parameters"], graph_dict["distances"])
                if normalize:
                    for dist in dists:
                        sim_dist_scaled.append(self.scale_distance(dist))

            if normalize:
                sim_dist_scaled = np.asarray(sim_dist_scaled)
            else:
                sim_dist_scaled = np.asarray(dists)

            # Use MAB arm selection to identify the best 'k' arms or summary statistics
            num_arms = sim_dist_scaled.shape[1]
            arms = range(num_arms)
            top_k_arms_idx = self.mab_variant.select(arms, self.k)
            top_k_distances = np.asarray([sim_dist_scaled[:, i] for i in top_k_arms_idx])
            top_k_distances = top_k_distances.transpose()

            # Take the norm to combine the distances, if more than one summary is used
            if top_k_distances.shape[1] > 1:
                combined_distance = [dask.delayed(np.linalg.norm)(scaled) for scaled in top_k_distances]
                result, = dask.compute(combined_distance)
            else:
                result = top_k_distances.ravel()

            # Accept/Reject
            for e, res in enumerate(result):
                if self.use_logger:
                    self.logger.debug("Bandits-ABC Rejection Sampling: trial parameter(s) = {}".format(params[e]))
                    self.logger.debug("Bandits-ABC Rejection Sampling: "
                                      "trial distance(s) = {}".format(sim_dist_scaled[e]))
                if res <= self.epsilon:
                    accepted_samples.append(params[e])
                    distances.append(dists[e])
                    accepted_count += 1
                    if self.use_logger:
                        self.logger.info("Bandits-ABC Rejection Sampling: accepted a new sample, "
                                         "total accepted samples = {0}".format(accepted_count))

            trial_count += batch_size

        self.results = {'accepted_samples': accepted_samples, 'distances': distances, 'accepted_count': accepted_count,
                        'trial_count': trial_count, 'inferred_parameters': np.mean(accepted_samples, axis=0)}
        return self.results
