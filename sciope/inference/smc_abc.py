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
from sciope.utilities.epsilonselectors import RelativeEpsilonSelector
from sciope.utilities.perturbationkernels.multivariate_normal import MultivariateNormalKernel

import numpy as np
import dask
from dask.distributed import futures_of, as_completed, wait
from dask import delayed
import pickle
import os
import glob


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

    Methods:
    * infer 					(perform parameter inference)
    """

    def __init__(self, data, sim, prior_function, problem_name=None,
                 perturbation_kernel=None,
                 summaries_function=bs.Burstiness().compute,
                 distance_function=euc.EuclideanDistance(),
                 summaries_divisor=None, use_logger=False,
                 max_sampling_iterations=np.Inf):

        self.name = 'SMC-ABC'
        super(SMCABC, self).__init__(self.name, data, sim, problem_name, use_logger)

        self.prior_function = prior_function
        self.summaries_function = summaries_function
        self.distance_function = distance_function
        self.summaries_divisor = summaries_divisor
        self.max_sampling_iterations = max_sampling_iterations
        if perturbation_kernel is not None:
            self.perturbation_kernel = perturbation_kernel
        else:
            self.perturbation_kernel = MultivariateNormalKernel(
                d=self.prior_function.get_dimension(),
                adapt=True)

        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Sequential Monte-Carlo Approximate Bayesian Computation initialized")

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

        # Check if problem name is specified, then check if the path is also specified
        if self.problem_name is not None:
            if path and os.path.isdir(path):
                print(f"Using the path: {path}")
                saved_runs_path = os.path.join(path, f'{self.problem_name}_saved_runs')
                saved_kernels_path = os.path.join(path, f'{self.problem_name}_saved_kernels')
                os.makedirs(saved_runs_path, exist_ok=True)
                os.makedirs(saved_kernels_path, exist_ok=True)
            else:
                print(f"Using the path: {os.getcwd()}")
                saved_runs_path = f'{self.problem_name}_saved_runs'
                saved_kernels_path = f'{self.problem_name}_saved_kernels'
                os.makedirs(saved_runs_path, exist_ok=True)
                os.makedirs(saved_kernels_path, exist_ok=True)
        else:
            if path and os.path.isdir(path):
                print(f"Using the path: {path}")
                saved_runs_path = os.path.join(path, 'saved_runs')
                saved_kernels_path = os.path.join(path, 'saved_kernels')
                os.makedirs(saved_runs_path, exist_ok=True)
                os.makedirs(saved_kernels_path, exist_ok=True)
            else:
                print(f"Using the path: {os.getcwd()}")
                saved_runs_path = 'saved_runs'
                saved_kernels_path = 'saved_kernels'
                os.makedirs(saved_runs_path, exist_ok=True)
                os.makedirs(saved_kernels_path, exist_ok=True)

        saved_kernels = sorted(glob.glob(os.path.join(saved_kernels_path, 'kernel_*.pkl')))
        saved_runs = sorted(glob.glob(os.path.join(saved_runs_path, 'smcabc_*.pkl')))

        def run_from_start(self):
            abc_history = []
            t = num_samples
            prior_function = self.prior_function

            tol, relative, terminate = eps_selector.get_initial_epsilon()  # Difference between the two function
            print("Determining initial population for round 0 using epsilon: {}".format(tol))
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

            population = np.vstack(abc_results['accepted_samples'])[:t]
            normalized_weights = np.ones(t) / t

            abc_history.append(abc_results)
            abc_history[0]['eps'] = tol  # add epsilon values for each round

            if saved_runs_path:
                file_path = os.path.join(saved_runs_path, f'smcabc_{0}.pkl')
                with open(file_path, 'wb') as f:
                    pickle.dump(abc_history, f)

            # SMC iterations
            round = 1
            max_rounds = eps_selector.max_rounds

            while round <= max_rounds:

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
                    # Run ABC on the next epsilon using the proposal prior
                    abc_instance = abc_inference.ABC(self.data, self.sim, new_prior,
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

                    # Compute importance weights for the new samples
                    if len(abc_results['accepted_samples']) == t:
                        new_samples = np.vstack(abc_results['accepted_samples'])[:t]

                        prior_weights = self.prior_function.pdf(new_samples)
                        kweights = self.perturbation_kernel.pdf(population, new_samples)

                        new_weights = prior_weights / np.sum(kweights * normalized_weights[:, np.newaxis], axis=0)
                        new_weights = new_weights / sum(new_weights)

                        population = new_samples
                        normalized_weights = new_weights

                        abc_history.append(abc_results)
                        abc_history[round]['eps'] = tol

                    # Save the perturbation kernel for the next round
                    if saved_kernels_path:
                        file_path_1 = os.path.join(saved_kernels_path, f'kernel_{round - 1}.pkl')
                        with open(file_path_1, 'wb') as f:
                            pickle.dump(self.perturbation_kernel, f)

                        # Dump abc_history to a text
                        with open(os.path.join(saved_runs_path,
                                               f"dump_saved_smc_abc_history_{round}_{self.problem_name}.txt"),
                                  "w") as f:
                            f.write(str(abc_history))

                        file_path = os.path.join(saved_runs_path, f'smcabc_{round}.pkl')
                        with open(file_path, 'wb') as f:
                            pickle.dump(abc_history, f)

                        round += 1

                except KeyboardInterrupt:
                    return abc_history
                except:
                    raise

                if resume:
                    if round == (max_rounds):
                        choice = input("Do you want to continue with 3 more rounds? (y/n): ")
                        if choice.lower() == 'y':
                            max_rounds += 3
                        else:
                            break

            return abc_history

        def run_from_particular_round(self, abc_history, round, population, kernel, max_rounds):
            print(f" Note: SMC_ABC will run for {max_rounds} rounds at a time")

            if resume:
                if round == (max_rounds):
                    choice = input("Do you want to continue with 3 more rounds? (y/n): ")
                    if choice.lower() == 'y':
                        max_rounds += 3
                    else:
                        return abc_history

            t = num_samples
            normalized_weights = np.ones(t) / t
            i = round
            while i < max_rounds:
                tol, relative, terminate = eps_selector.get_epsilon(i, abc_history)
                print("Determining initial population for round {} using epsilon: {}".format(i, tol))

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
                    # Run ABC on the next epsilon using the proposal prior
                    abc_instance = abc_inference.ABC(self.data, self.sim, new_prior,
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

                    # Compute importance weights for the new samples
                    if len(abc_results['accepted_samples']) == t:
                        new_samples = np.vstack(abc_results['accepted_samples'])[:t]

                        prior_weights = self.prior_function.pdf(new_samples)
                        kweights = perturbation_kernel.pdf(population, new_samples)

                        new_weights = prior_weights / np.sum(kweights * normalized_weights[:, np.newaxis], axis=0)
                        new_weights = new_weights / sum(new_weights)

                        population = new_samples
                        normalized_weights = new_weights

                        abc_history.append(abc_results)
                        abc_history[i]['eps'] = tol

                    # Save the perturbation kernel for the next round
                    file_path_1 = os.path.join(saved_kernels_path, f'kernel_{i}.pkl')
                    with open(file_path_1, 'wb') as f:
                        pickle.dump(self.perturbation_kernel, f)

                    # Save the state in pkl file after each round
                    file_path = os.path.join(saved_runs_path, f'smcabc_{i}.pkl')
                    with open(file_path, 'wb') as f:
                        pickle.dump(abc_history, f)

                    i = i + 1

                    if resume:
                        if i == (max_rounds):
                            choice = input("Do you want to continue with 3 more rounds? (y/n): ")
                            if choice.lower() == 'y':
                                max_rounds += 3
                            else:
                                return abc_history

                except KeyboardInterrupt:
                    return abc_history
                except:
                    raise

            return abc_history

        if round >= 0:
            if saved_runs:
                # First check if there are saved files
                print(f"Found {len(saved_runs)} saved runs:")
                for i, file_path in enumerate(saved_runs):
                    with open(file_path, 'rb') as f:
                        saved_history = pickle.load(f)
                    print(f"{i}. Round {i} - {file_path}")

                if 0 <= round <= len(saved_runs):

                    if round > 0 and round <= len(saved_runs):
                        t = num_samples
                        population = np.vstack(saved_history[round - 1]['accepted_samples'])[:t]
                        loaded_eps = saved_history[round - 1]['eps']
                        file_path_ = os.path.join(saved_kernels_path, f'kernel_{round - 1}.pkl')

                        # file_path_ = os.path.join(f'{self.problem_name}_saved_kernels', f'kernel_{round - 1}.pkl')
                        with open(file_path_, 'rb') as f:
                            kernel = pickle.load(f)

                        print(f'Starting from round {round} with epsilon {loaded_eps}')

                        abc_history = run_from_particular_round(self, saved_history, round, population, kernel,
                                                                max_rounds=eps_selector.max_rounds)

                        return abc_history

                    elif round == 0:
                        print('No round is specified, Starting from round 0')
                        dir_path_1 = saved_runs_path
                        dir_path_2 = saved_kernels_path

                        # loop over the files in the directory and delete them
                        for file_name in os.listdir(dir_path_1):
                            file_path = os.path.join(dir_path_1, file_name)
                            try:
                                if os.path.isfile(file_path):
                                    os.unlink(file_path)
                                    print(f"Deleted {file_path}")
                            except Exception as e:
                                print(f"Failed to delete {file_path}. Reason: {e}")

                        for file_name in os.listdir(dir_path_2):
                            file_path = os.path.join(dir_path_2, file_name)
                            try:
                                if os.path.isfile(file_path):
                                    os.unlink(file_path)
                                    print(f"Deleted {file_path}")
                            except Exception as e:
                                print(f"Failed to delete {file_path}. Reason: {e}")
                        # After deleting all the saved file, start a fresh run
                        abc_history = run_from_start(self)
                        return abc_history
                    else:
                        print(f"No saved {round} round exist")



                else:
                    print("Please enter a valid round number")
            else:
                print('No saved runs found')
                abc_history = run_from_start(self)
                return abc_history

