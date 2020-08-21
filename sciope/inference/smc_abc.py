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
Sequential Monte-Carlo Approximate Bayesian Computation (SMC-ABC)
"""

# Imports
from sciope.inference.abc_inference import ABC
from sciope.inference.inference_base import InferenceBase
from sciope.inference import abc_inference
from sciope.core import core
from sciope.utilities.distancefunctions import euclidean as euc
from sciope.utilities.summarystats import burstiness as bs
from sciope.utilities.housekeeping import sciope_logger as ml
from sciope.utilities.priors.prior_base import PriorBase

import numpy as np
import dask
from dask.distributed import futures_of, as_completed, wait
from dask import delayed


class PerturbationPrior(PriorBase):

    def __init__(self, ref_prior, samples, normalized_weights, perturbation_kernel,
                 use_logger=False):

        self.name = 'Perturbation Prior'
        self.ref_prior = ref_prior
        self.samples = samples
        self.normalized_weights = normalized_weights
        self.perturbation_kernel = perturbation_kernel
        super(PerturbationPrior, self).__init__(self.name, use_logger)

    def draw(self, n=1, chunk_size=1):

        assert n >= chunk_size

        generated_samples = []
        m = n % chunk_size
        if m > 0:
            generated_samples.append(self._weighted_draw_perturb(m))

        for i in range(0, n - m, chunk_size):
            generated_samples.append(self._weighted_draw_perturb(chunk_size))

        return generated_samples

    @delayed
    def _weighted_draw_perturb(self, m):
        idxs = np.random.choice(self.samples.shape[0], m,
                                p=self.normalized_weights)
        s0 = [self.samples[idx] for idx in idxs]
        s = []
        for z in s0:
            accepted = False
            while not accepted:
                sz = self.perturbation_kernel.rvs(z)
                if self.ref_prior.pdf(sz) > 0:
                    accepted = True
                    s.append(sz)

        return np.asarray(s)


class SMCABC(InferenceBase):
    """
    SMC - Approximate Bayesian Computation

    Properties/variables:
    * data                      (observed / fixed data)
    * sim   					(simulator function handle)
    * prior_function			(prior over the simulator parameters)
    * perturbation_kernel       (kernel for perturbing parameters samples)
    * epsilons 	    			(sequence of tolerance bounds)
    * summaries_function    	(summary statistics calculation function)
    * distance_function         (function calculating deviation between simulated statistics and observed statistics)
    * summaries_divisor         (numpy array of maxima - used for normalizing summary statistic values)
    * use_logger    			(whether logging is enabled or disabled)

    Methods:
    * infer 					(perform parameter inference)
    """

    def __init__(self, data, sim, prior_function, perturbation_kernel,
                 epsilons=[1], summaries_function=bs.Burstiness(),
                 distance_function=euc.EuclideanDistance(),
                 summaries_divisor=None, use_logger=False):

        self.name = 'SMC-ABC'
        super(SMCABC, self).__init__(self.name, data, sim, use_logger)

        self.prior_function = prior_function
        self.summaries_function = summaries_function
        self.epsilons = epsilons
        self.distance_function = distance_function
        self.summaries_divisor = summaries_divisor
        self.perturbation_kernel = perturbation_kernel

        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Sequential Monte-Carlo Approximate Bayesian Computation initialized")

    def infer(self, num_samples, batch_size, chunk_size=10, ensemble_size=1, normalize=False):
        """
        Performs SMC-ABC.

        Parameters
        ----------
        num_samples : int
            The number of required accepted samples
        batch_size : int
            The batch size of samples for performing rejection sampling
        chunk_size : int
            The partition size when splitting the fixed data. For avoiding many individual tasks
            in dask if the data is large. Default 10.
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

        t = num_samples

        prior_function = self.prior_function

        # Generate an initial population from the first epsilon
        abc_instance = abc_inference.ABC(self.data, self.sim, prior_function,
                                         epsilon=self.epsilons[0],
                                         summaries_function=self.summaries_function,
                                         distance_function=self.distance_function,
                                         summaries_divisor=self.summaries_divisor,
                                         use_logger=self.use_logger)

        print("Starting epsilon={}".format(self.epsilons[0]))
        abc_instance.compute_fixed_mean(chunk_size=chunk_size)
        abc_results = abc_instance.infer(num_samples=t, batch_size=batch_size, chunk_size=chunk_size,
                                         normalize=normalize)

        final_results = abc_results
        population = np.vstack(abc_results['accepted_samples'])[:t]
        normalized_weights = np.ones(t) / t
        d = population.shape[1]

        # SMC iterations
        for eps in self.epsilons[1:]:
            print("Starting epsilon={}".format(eps))
            if self.use_logger:
                self.logger.info("Starting epsilon={}".format(eps))

            # Adapt the kernel based on the current population
            self.perturbation_kernel.adapt(population)

            # Generate a proposal prior based on the population
            new_prior = PerturbationPrior(self.prior_function, population, normalized_weights, self.perturbation_kernel)

            try:
                # Run ABC on the next epsilon using the proposal prior
                abc_instance = abc_inference.ABC(self.data, self.sim, new_prior,
                                                 epsilon=eps, summaries_function=self.summaries_function,
                                                 distance_function=self.distance_function,
                                                 summaries_divisor=self.summaries_divisor,
                                                 use_logger=self.use_logger)
                abc_instance.compute_fixed_mean(chunk_size=chunk_size)
                abc_results = abc_instance.infer(num_samples=t,
                                                 batch_size=batch_size,
                                                 chunk_size=chunk_size,
                                                 normalize=normalize)
                new_samples = np.vstack(abc_results['accepted_samples'])[:t]

                # Compute importance weights for the new samples
                prior_weights = self.prior_function.pdf(new_samples)
                kweights = self.perturbation_kernel.pdf(population, new_samples)

                new_weights = prior_weights / np.sum(kweights * normalized_weights[:, np.newaxis], axis=0)
                new_weights = new_weights / sum(new_weights)

                population = new_samples
                normalized_weights = new_weights
                final_results = abc_results
            except KeyboardInterrupt:
                return final_results
            except:
                raise

        return final_results
