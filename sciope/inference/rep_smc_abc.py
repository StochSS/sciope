# Copyright 2019 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
Replenisment Sequential Monte-Carlo Approximate Bayesian Computation (SMC-ABC)
"""

# Imports
from sciope.inference.abc_inference import ABC
from sciope.inference.inference_base import InferenceBase
from sciope.inference import abc_inference
from sciope.core import core
from sciope.utilities.distancefunctions import euclidean as euc
from sciope.utilities.summarystats import identity
from sciope.utilities.housekeeping import sciope_logger as ml
from sciope.utilities.priors.prior_base import PriorBase
from sciope.utilities.epsilonselectors import RelativeEpsilonSelector
from sciope.utilities.perturbationkernels.multivariate_normal import MultivariateNormalKernel

import numpy as np
import dask
from dask.distributed import futures_of, as_completed, wait
from dask import delayed

class ReplenishmentSMCABC(InferenceBase):
    """
    Replenishment SMC - Approximate Bayesian Computation

    Properties/variables:
    * data                      (observed / fixed data)
    * sim   					(simulator function handle)
    * prior_function			(prior over the simulator parameters)
    * perturbation_kernel       (kernel for perturbing parameters samples)
    * summaries_function    	(summary statistics calculation function)
    * distance_function         (function calculating deviation between simulated statistics and observed statistics)
    * summaries_divisor         (numpy array of maxima - used for normalizing summary statistic values)
    * use_logger    			(whether logging is enabled or disabled)

    Methods:
    * infer 					(perform parameter inference)
    """

    def __init__(self, data, sim, prior_function, 
                 perturbation_kernel=None,
                 summaries_function=identity.Identity().compute,
                 distance_function=euc.EuclideanDistance(),
                 summaries_divisor=None, use_logger=False):

        self.name = 'SMC-ABC'
        super(ReplenishmentSMCABC, self).__init__(self.name, data, sim, use_logger)

        self.prior_function = prior_function
        self.summaries_function = summaries_function
        self.distance_function = distance_function
        self.summaries_divisor = summaries_divisor
        if perturbation_kernel is not None:
            self.perturbation_kernel = perturbation_kernel
        else:
            self.perturbation_kernel = MultivariateNormalKernel(
                    d = self.prior_function.get_dimension(),
                    adapt = True)

        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Sequential Monte-Carlo Approximate Bayesian Computation initialized")

    def _simulate_N(self, prior, N, params = None):

        if params is None:
            params_delayed = prior.draw(N)
            params, = dask.compute(params_delayed, num_workers = 10)
            params = np.vstack(params).reshape(N, -1)

        results = []
        for i in range(params.shape[0]):
            current_param = params[i,:]
            lazy_results = dask.delayed(self._simulate_distance)(current_param)
            results.append(lazy_results)

        computed_results, = dask.compute(results)
        dists, c = list(zip(*computed_results))

        return np.asarray(dists), params, np.sum(c)

    def _simulate_distance(self, param):
        r, c = self.sim(param)
        if r is None:
            return np.inf, c
        else:
            res = self.summaries_function(r)
            res_obs = self.summaries_function(self.data)
            return self.distance_function.compute(res_obs, res), c

    def infer(self, num_samples, alpha = 0.5, initial_population = None, R_trial = 10, c = 0.01, p_min = 0.05):
        """Performs SMC-ABC.

        Parameters
        ----------
        num_samples : int
            The number of required accepted samples
        alpha : float
            Culling percentage
        R_trial : int
            Number of perturbs per replenishment

        Returns
        -------
        dict
            Keys
            'accepted_samples: The accepted parameter values',
            'distances: Accepted distance values'
        """
        def perturb_resample(p, cur_distance, R, tol):
            p_acc = 0
            n_moves = 0
            new_distance = cur_distance
            ssa_count = 0
            N_count  = 0
            for _ in range(R):
                z = self.perturbation_kernel.rvs(p)
                num = prior.pdf(z)
                # Metropolis Perturbation
                if num > 0:
                    num = num * self.perturbation_kernel.pdf(p.reshape(1,-1), z.reshape(1,-1))[0,0,]
                    dnm = prior.pdf(p) * self.perturbation_kernel.pdf(z.reshape(1,-1), p.reshape(1,-1))[0,0]
                    weight = num/dnm
                    if np.random.rand() < min(1, weight):
                        distance, c = self._simulate_distance(z)
                        ssa_count += c
                        N_count += 1
                        if distance <= tol:
                            n_moves += 1
                            p = z
                            new_distance = distance
                            p_acc = p_acc + (1/(R * Na))
            return p, new_distance, p_acc, n_moves, ssa_count, N_count

        prior = self.prior_function
        Na = round(num_samples * alpha)

        total_ssa = 0
        total_N = 0
        # Sample N particles
        distances, population, ssa_count = self._simulate_N(prior, num_samples, params = initial_population)

        total_ssa = ssa_count
        total_N = num_samples
        terminate = False
        while not terminate:
            try:
                iter_ssa_count = 0
                iter_N_count = 0
                # Sort population by distance
                sorted_idxs = np.argsort(distances)
                population = population[sorted_idxs,:]
                distances = distances[sorted_idxs]
                # Cull the last Na
                population = population[:Na]
                distances = distances[:Na]
                tol = distances[-1]

                # Resample with replacement
                resampled_idxs = np.random.choice(Na, num_samples - Na)
                population = np.vstack([population, population[resampled_idxs]])
                distances = np.concatenate([distances, distances[resampled_idxs]])
                # Adapt transition kernel using samples
                self.perturbation_kernel.adapt(population)

                perturbed_ps = []
                # Perturb each of the resampled values
                for i in range(Na, num_samples):
                    perturbed_ps.append(dask.delayed(perturb_resample)(population[i,:], distances[i], R_trial, tol))
                res, = dask.compute(perturbed_ps)

                updated_ps, updated_distances, update_p_accs, N_accs, ssa_count, N_count = list(zip(*res))
                population[Na:] = np.vstack(updated_ps)
                distances[Na:] = np.asarray(updated_distances)
                p_acc = np.sum(update_p_accs)
                N_acc = np.sum(N_accs)
                iter_ssa_count += np.sum(ssa_count)
                iter_N_count += np.sum(N_count)

                R = int(round(np.log(c) / np.log(1 - p_acc)))

                perturbed_ps = []
                for i in range(Na, num_samples):
                    perturbed_ps.append(dask.delayed(perturb_resample)(population[i,:], distances[i], R, tol))
                res, = dask.compute(perturbed_ps)

                updated_ps, updated_distances, update_p_accs, N_accs, ssa_count, N_count = list(zip(*res))
                population[Na:] = np.vstack(updated_ps)
                distances[Na:] = np.asarray(updated_distances)
                p_acc += p_acc + np.sum(update_p_accs)
                N_acc += np.sum(N_accs)
                iter_ssa_count += np.sum(ssa_count)
                iter_N_count += np.sum(N_count)
                total_ssa += iter_ssa_count
                total_N += iter_N_count
                print("Tol : {}, R : {}, N_acc : {}, p_acc : {}, iter_ssa_count : {}, iter_N_count : {}".format(tol, R, N_acc, p_acc, iter_ssa_count, iter_N_count))
                if p_acc < p_min:
                    terminate = True
            except KeyboardInterrupt:
                return {'accepted_samples' : population, 'distances' : distances, 'ssa_count' : total_ssa, 'N_count' : total_N}
            except:
                raise

        return {'accepted_samples' : population, 'distances' : distances, 'ssa_count' : total_ssa, 'N_count' : total_N}
