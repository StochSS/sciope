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
                 summaries_function=identity.Identity(),
                 distance_function=euc.EuclideanDistance(),
                 summaries_divisor=None, use_logger=False):
        """Replenishment SMC-ABC implementation.

        Parameters
        ----------
        data : nd-array
            the observed / fixed dataset
        sim : Callable[[nd-array], nd-array]
            the simulator function
        prior_function : sciope.utilities.priors object
            the prior function generating candidate samples
        perturbation_kernel : sciope.utilities.perturbationkernels object, optional
            kernel to perturb samples
        summaries_function : sciope.utilities.summarystats object, optional
            function calculating summary stats over simulated results
        distance_function : sciope.utilities.distancefunction, optional
            distance function operating over summary statistics
        use_logger : bool
            enable/disable logging
        """

        self.name = 'Replenisment-SMC-ABC'
        super(ReplenishmentSMCABC, self).__init__(self.name, data, sim, use_logger)

        self.prior_function = prior_function
        self.summaries_function = summaries_function
        self.distance_function = distance_function.compute
        self.summaries_divisor = summaries_divisor
        if perturbation_kernel is not None:
            self.perturbation_kernel = perturbation_kernel
        else:
            self.perturbation_kernel = MultivariateNormalKernel(
                d=self.prior_function.get_dimension(),
                adapt=True)

        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Replenisment Sequential Monte-Carlo \
                              Approximate Bayesian Computation initialized")

    def compute_fixed_mean(self, chunk_size):
        stats_mean = core.get_fixed_mean(self.data, self.summaries_function, chunk_size)
        self.fixed_mean = stats_mean.compute()
        del stats_mean

    @dask.delayed
    def _perturb_resample(self, param, current_distance, n_perturbations, tol):
        p_acc = 0
        n_successful_moves = 0
        new_distance = current_distance

        for _ in range(n_perturbations):
            proposal = self.perturbation_kernel.rvs(param)
            nume = self.prior_function.pdf(proposal)

            # Metropolis Perturbation
            if nume > 0:
                nume = nume * self.perturbation_kernel.pdf(param.reshape(1, -1), proposal.reshape(1, -1))[0, 0,]
                dnm = self.prior_function.pdf(param) * \
                      self.perturbation_kernel.pdf(proposal.reshape(1, -1), param.reshape(1, -1))[0, 0]

                weight = nume / dnm
                if np.random.rand() < min(1, weight):
                    proposal_ss = self.summaries_function(self.sim(proposal))
                    distance = self.distance_function(self.fixed_mean, proposal_ss)
                    if distance <= tol:
                        n_successful_moves += 1
                        param = proposal
                        new_distance = distance

                        # Update estimate of movement probability
                        p_acc += 1 / n_perturbations

        return param, new_distance, p_acc, n_successful_moves

    def infer(self, num_samples, alpha=0.5, R_trial=10, c=0.01, p_min=0.05, batch_size=10, chunk_size=1):
        """Performs SMC-ABC.

        Parameters
        ----------
        num_samples : int
            The number of required accepted samples
        alpha : float
            Culling percentage
        R_trial : int
            Number of perturbs per replenishment to estimate probability
        c : float
            Sensitivity for more perturbations
        p_min : float
            Termination condition as a probability of a successul perturbation

        Returns
        -------
        dict
            Keys
            'accepted_samples: The accepted parameter values',
            'distances: Accepted distance values'
        """

        assert hasattr(self, "fixed_mean"), "Please call compute_fixed_mean before infer"

        # Get the dask graph and add another distances task to it
        graph_dict = core.get_graph_chunked(self.prior_function.draw, self.sim, self.summaries_function,
                                            batch_size, chunk_size)
        dist_func = lambda x: self.distance_function(self.fixed_mean, x)
        graph_dict["distances"] = core.get_distance(dist_func, graph_dict["summarystats"], chunked=True)

        # Culling Cutoff
        n_cull = round(alpha * num_samples)

        # Draw the initial population and compute distances
        population, distances = dask.compute(graph_dict['parameters'], graph_dict['distances'])
        population = core._reshape_chunks(population)
        distances = core._reshape_chunks(distances)

        while population.shape[0] < num_samples:
            params, dists = dask.compute(graph_dict["parameters"], graph_dict["distances"])
            params = core._reshape_chunks(params)
            dists = core._reshape_chunks(dists)
            population = np.vstack([population, params])
            distances = np.vstack([distances, dists])

        population = population[:num_samples]
        distances = distances[:num_samples, 0]

        terminate = False
        while not terminate:

            try:
                # Sort population by distance
                sorted_idxs = np.argsort(distances)
                population = population[sorted_idxs]
                distances = distances[sorted_idxs]

                # Cull the last Na
                population = population[:n_cull]
                distances = distances[:n_cull]
                tol = distances[-1]

                # Resample with replacement to replenish in the population
                resampled_idxs = np.random.choice(n_cull, num_samples - n_cull)
                population = np.vstack([population, population[resampled_idxs]])
                distances = np.concatenate([distances, distances[resampled_idxs]])

                # Adapt transition kernel using the new population
                self.perturbation_kernel.adapt(population)

                # For each replenished value, perturb and resample a few time
                # to get an idea of how easy it is to move to a lower distance
                perturb_tasks = []
                for i in range(n_cull, num_samples):
                    perturb_tasks.append(self._perturb_resample(population[i, :], distances[i], R_trial, tol))
                res, = dask.compute(perturb_tasks)

                # Update the population with the perturbed population
                updated_ps, updated_distances, update_p_accs, N_accs = list(zip(*res))

                population[n_cull:] = np.vstack(updated_ps)
                distances[n_cull:] = np.asarray(updated_distances)

                # Update metrics from the trial to estimate the probability
                # of a move to assess convergence and decide how many more
                # perturbation attempts to make
                p_acc = np.sum(update_p_accs) / (num_samples - n_cull)
                N_acc = np.sum(N_accs)

                R = int(round(np.log(c) / np.log(1 - p_acc)))

                # Perturb again with better estimate
                perturb_tasks = []
                for i in range(n_cull, num_samples):
                    perturb_tasks.append(self._perturb_resample(population[i, :], distances[i], R - R_trial, tol))
                res, = dask.compute(perturb_tasks)

                updated_ps, updated_distances, update_p_accs, N_accs = list(zip(*res))

                population[n_cull:] = np.vstack(updated_ps)
                distances[n_cull:] = np.asarray(updated_distances)

                p_acc += np.sum(update_p_accs) / (num_samples - n_cull)
                N_acc += np.sum(N_accs)

                print("Tol : {}, R : {}, p_acc : {}".format(tol, R, p_acc))
                if p_acc < p_min:
                    terminate = True
            except KeyboardInterrupt:
                return {'accepted_samples': population, 'distances': distances}
            except:
                raise

        return {'accepted_samples': population, 'distances': distances}
