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

import glob
import os
import pickle

import dask
import numpy as np
from dask import delayed
from dask.distributed import futures_of, as_completed, wait
from sciope.core import core
from sciope.inference import abc_inference
# Imports
from sciope.inference.abc_inference import ABC
from sciope.inference.inference_base import InferenceBase
from sciope.utilities.distancefunctions import euclidean as euc
from sciope.utilities.epsilonselectors import RelativeEpsilonSelector
from sciope.utilities.housekeeping import sciope_logger as ml
from sciope.utilities.perturbationkernels.multivariate_normal import MultivariateNormalKernel
from sciope.utilities.priors.prior_base import PriorBase
from sciope.utilities.summarystats import burstiness as bs
from sciope.visualize.inference_results import InferenceResults, InferenceRound


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
    * summaries_function    	(summary statistics calculation function)
    * distance_function         (function calculating deviation between simulated statistics and observed statistics)
    * summaries_divisor         (numpy array of maxima - used for normalizing summary statistic values)
    * use_logger    			(whether logging is enabled or disabled)
    * parameters                (dict of parameters in the form of {name: value} - used for generating result objects)

    Methods:
    * infer 					(perform parameter inference)
    """

    def __init__(self, data, sim, prior_function, problem_name=None,
                 perturbation_kernel=None,
                 summaries_function=bs.Burstiness().compute,
                 distance_function=euc.EuclideanDistance(),
                 summaries_divisor=None, use_logger=False, parameters=None,
                 max_sampling_iterations=np.Inf):

        self.name = 'SMC-ABC'
        super(SMCABC, self).__init__(self.name, data, sim, problem_name, use_logger)

        self.prior_function = prior_function
        self.summaries_function = summaries_function
        self.distance_function = distance_function
        self.summaries_divisor = summaries_divisor
        self.max_sampling_iterations = max_sampling_iterations
        self.parameters = parameters
        if perturbation_kernel is not None:
            self.perturbation_kernel = perturbation_kernel
        else:
            self.perturbation_kernel = MultivariateNormalKernel(
                d=self.prior_function.get_dimension(),
                adapt=True)

        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Sequential Monte-Carlo Approximate Bayesian Computation initialized")

    def create_directories(self, path=None):

        if self.problem_name is not None:
            directory_prefix = f'{self.problem_name}_'
        else:
            directory_prefix = ''

        if path:
            path = os.path.normpath(path)
            if os.path.isfile(path):
                raise ValueError(f"The provided path '{path}' points to a file, not a directory.")
        if path and os.path.isdir(path):
            print(f"Using the path: {path}")
        else:
            if path:
                print(
                    f"WARNING: The provided path '{path}' does not exist. Using the current working directory instead.")
            else:
                print(f"No path provided. Using the current working directory.")
            path = os.getcwd()

        saved_runs_path = os.path.join(path, f'{directory_prefix}saved_runs')
        saved_kernels_path = os.path.join(path, f'{directory_prefix}saved_kernels')

        os.makedirs(saved_runs_path, exist_ok=True)
        os.makedirs(saved_kernels_path, exist_ok=True)

        return saved_runs_path, saved_kernels_path

    def delete_files_in_directory(self, directory_path):

        """ Helper function to delete saved files, if we start from round 0"""

        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    def _delete_rounds(self, round, max_rounds, saved_runs_path, saved_kernels_path):

        """ Helper function to delete saved files for rounds and kernels
         starting from the given round up to max_rounds """

        for j in range(round, max_rounds + 1):
            file_path = os.path.join(saved_runs_path, f'smcabc_{j}.pkl')
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted {file_path}")

            kernel_path = os.path.join(saved_kernels_path, f'kernel_{j}.pkl')
            if os.path.isfile(kernel_path):
                os.unlink(kernel_path)
                print(f"Deleted {kernel_path}")

    def save_state_and_kernel(self, abc_history, perturbation_kernel,
                              saved_kernels_path, saved_runs_path, round):

        # Save the perturbation kernel for the next round
        file_path_1 = os.path.join(saved_kernels_path, f'kernel_{round}.pkl')
        with open(file_path_1, 'wb') as f:
            pickle.dump(perturbation_kernel, f)

        # Save the state in pkl file after each round
        file_path = os.path.join(saved_runs_path, f'smcabc_{round}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(abc_history, f)

    def _abc_instance(self, tol, t, prior_function,
                      chunk_size, batch_size, relative, abc_history):

        """Helper function which creates an ABC instance"""

        abc_instance = abc_inference.ABC(self.data, self.sim, prior_function,
                                         epsilon=tol,
                                         summaries_function=self.summaries_function,
                                         distance_function=self.distance_function,
                                         summaries_divisor=self.summaries_divisor,
                                         use_logger=self.use_logger,
                                         max_sampling_iterations=self.max_sampling_iterations)

        abc_instance.compute_fixed_mean(chunk_size=chunk_size)
        abc_results = abc_instance.infer(num_samples=t,
                                         batch_size=batch_size,
                                         chunk_size=chunk_size,
                                         normalize=relative)

        return abc_results

    def _run_iteration(self, tol, population, normalized_weights, perturbation_kernel,
                       new_prior, chunk_size, t, batch_size, relative, abc_history, round):

        """ Helper Function to run a single iteration of SMC-ABC """

        abc_results = self._abc_instance(tol, t, new_prior, chunk_size,
                                         batch_size, relative, abc_history)

        # Compute importance weights for the new samples
        new_samples = np.vstack(abc_results['accepted_samples'])[:t]
        prior_weights = self.prior_function.pdf(new_samples)
        kweights = perturbation_kernel.pdf(population, new_samples)

        new_weights = prior_weights / np.sum(kweights * normalized_weights[:, np.newaxis], axis=0)
        new_weights = new_weights / sum(new_weights)

        population = new_samples
        normalized_weights = new_weights

        if self.parameters is not None:
            inference_round = InferenceRound.build_from_inference_round(abc_results,
                                                                        list(self.parameters.keys()))
            abc_results.update(
                inference_round)

        abc_history.append(abc_results)
        abc_history[round]['eps'] = tol

        return abc_results, abc_history

    def run_smc_from_start(self, num_samples, batch_size, chunk_size, saved_kernels_path,
                           saved_runs_path, eps_selector, resume,
                           ensemble_size=1):

        """ Run ABC-SMC from start i.e Round 0"""

        abc_history = []
        t = num_samples
        prior_function = self.prior_function

        tol, relative, terminate = eps_selector.get_initial_epsilon()
        print("Determining initial population for round 0 using epsilon: {}".format(tol))

        abc_results = self._abc_instance(tol, t, prior_function, chunk_size,
                                         batch_size, relative, abc_history)

        population = np.vstack(abc_results['accepted_samples'])[:t]
        normalized_weights = np.ones(t) / t

        if self.parameters is not None:
            inference_round = InferenceRound.build_from_inference_round(abc_results, list(self.parameters.keys()))
            abc_results.update(
                inference_round)  # Update the abc_results with the additional keys and values from inference_round

        abc_history.append(abc_results)
        abc_history[0]['eps'] = tol

        if saved_runs_path:
            file_path = os.path.join(saved_runs_path, f'smcabc_{0}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(abc_history, f)

        # SMC iterations
        round = 1

        while round < eps_selector.max_rounds:
            tol, relative, terminate = eps_selector.get_epsilon(round, abc_history)

            print("Determining initial population for round {} using epsilon: {}".format(round, tol))
            if self.use_logger:
                self.logger.info("Starting epsilon = {}".format(tol))

            # Adapt the kernel based on the current population
            self.perturbation_kernel.adapt(population)

            # Generate a proposal prior based on the population
            new_prior = PerturbationPrior(self.prior_function,
                                          population,
                                          normalized_weights,
                                          self.perturbation_kernel)

            try:
                abc_results, abc_history = self._run_iteration(tol, population, normalized_weights,
                                                               self.perturbation_kernel,
                                                               new_prior, chunk_size, t,
                                                               batch_size, relative, abc_history, round)

                self.save_state_and_kernel(abc_history, self.perturbation_kernel,
                                           saved_kernels_path, saved_runs_path, round)

                round += 1

            except KeyboardInterrupt:
                if self.parameters is None:
                    return abc_history

                return InferenceResults(
                    abc_history, self.parameters, [self.prior_function.lb, self.prior_function.ub]
                )
            except:
                raise

            if resume:
                if round == (eps_selector.max_rounds):
                    choice = input("Do you want to continue with 3 more rounds? (y/n): ")
                    if choice.lower() == 'y':
                        eps_selector.max_rounds += 3

        if self.parameters is None:
            return abc_history

        return InferenceResults(
            abc_history, self.parameters, [self.prior_function.lb, self.prior_function.ub]
        )

    def run_smc_from_particular_round(self, abc_history, batch_size, chunk_size,
                                      eps_selector, round, population, kernel,
                                      resume, num_samples, saved_kernels_path,
                                      saved_runs_path, ensemble_size=1):

        """ Run SMC-ABC from a saved round """

        # Delete redundant saved files
        self._delete_rounds(round, eps_selector.max_rounds, saved_runs_path, saved_kernels_path)

        if resume:
            if round == (eps_selector.max_rounds):
                choice = input("Do you want to continue with 3 more rounds? (y/n): ")
                if choice.lower() == 'y':
                    eps_selector.max_rounds += 3

        t = num_samples
        normalized_weights = np.ones(t) / t

        while round < eps_selector.max_rounds:
            tol, relative, terminate = eps_selector.get_epsilon(round, abc_history)
            print("Determining initial population for round {} using epsilon: {}".format(round, tol))

            if self.use_logger:
                self.logger.info("Starting epsilon = {}".format(tol))

            # Adapt the kernel based on the current population
            perturbation_kernel = kernel
            perturbation_kernel.adapt(population)

            # Generate a proposal prior based on the population
            new_prior = PerturbationPrior(self.prior_function,
                                          population,
                                          normalized_weights,
                                          perturbation_kernel)

            try:
                abc_results, abc_history = self._run_iteration(tol, population, normalized_weights,
                                                               self.perturbation_kernel,
                                                               new_prior, chunk_size, t,
                                                               batch_size, relative, abc_history, round)

                self.save_state_and_kernel(abc_history, perturbation_kernel,
                                           saved_kernels_path, saved_runs_path, round)

                round += 1

                if resume:
                    if round == (eps_selector.max_rounds):
                        choice = input("Do you want to continue with 3 more rounds? (y/n): ")
                        if choice.lower() == 'y':
                            eps_selector.max_rounds += 3

            except KeyboardInterrupt:
                if self.parameters is None:
                    return abc_history
                return InferenceResults(
                    abc_history, self.parameters, [self.prior_function.lb, self.prior_function.ub]
                )
            except:
                raise

        if self.parameters is None:
            return abc_history
        return InferenceResults(
            abc_history, self.parameters, [self.prior_function.lb, self.prior_function.ub]
        )

    def infer(self, num_samples, batch_size,
              eps_selector=RelativeEpsilonSelector(20), chunk_size=10,
              ensemble_size=1, round=0, path=None, resume=False):

        """Performs SMC-ABC.

        Parameters
        ----------
        num_samples : int
            The number of required accepted samples
        batch_size : int
            The batch size of samples for performing rejection sampling
        eps_selector : EpsilonSelector
            The epsilon selector to determine the sequence of epsilons
        chunk_size : int
            The partition size when splitting the fixed data. For avoiding many individual tasks
            in dask if the data is large. Default 10.
        ensemble_size : int
            In case we have an ensemble of responses
        normalize : bool
            Whether summary statistics should be normalized and epsilon be interpreted as a percentage
        round : int
            Specify from which round SMC-ABC will start from. If no saved files exist it will run from round 0
        resume: bool
            Prompt the user to choose if they want to continue for three additional rounds. Default: False
        Path : str
            Path variable to store kernels and results of each round. If no path is provided, the current working directory will be used

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

        saved_runs_path, saved_kernels_path = self.create_directories(path)

        # Sorting the files
        saved_kernels = sorted(glob.glob(os.path.join(saved_kernels_path, 'kernel_*.pkl')))
        saved_runs = sorted(glob.glob(os.path.join(saved_runs_path, 'smcabc_*.pkl')))

        if round < 0:
            print("Invalid Round Number")

        elif not saved_runs:
            print('No saved runs found')
            abc_history = self.run_smc_from_start(num_samples, batch_size, chunk_size,
                                                  saved_kernels_path, saved_runs_path,
                                                  eps_selector, resume)
            return abc_history
        else:
            print(f"Found {len(saved_runs)} saved runs:")
            for i, file_path in enumerate(saved_runs):
                if os.path.getsize(file_path) > 0:
                    with open(file_path, 'rb') as f:
                        saved_history = pickle.load(f)
                        print(f"{i}. Round {i} - {file_path}")
                else:
                    print(f"Skipping empty file: {file_path}")

            if 0 <= round <= len(saved_runs):

                if round > 0:
                    t = num_samples
                    population = np.vstack(saved_history[round - 1]['accepted_samples'])[:t]
                    loaded_eps = saved_history[round - 1]['eps']
                    file_path_ = os.path.join(saved_kernels_path, f'kernel_{round - 1}.pkl')

                    with open(file_path_, 'rb') as f:
                        kernel = pickle.load(f)
                    saved_history = saved_history[:round]
                    print(f'Starting from round {round}')
                    return self.run_smc_from_particular_round(saved_history, batch_size, chunk_size,
                                                              eps_selector, round, population, kernel,
                                                              resume, num_samples,
                                                              saved_kernels_path, saved_runs_path,
                                                              ensemble_size=1)
                else:
                    print('No round is specified, Starting from round 0')
                    # After deleting all the saved files, start a fresh run
                    self.delete_files_in_directory(saved_runs_path)
                    self.delete_files_in_directory(saved_kernels_path)
                    return self.run_smc_from_start(num_samples, batch_size, chunk_size,
                                                   saved_kernels_path, saved_runs_path,
                                                   eps_selector, resume)

            else:
                print("Please enter a valid round number")