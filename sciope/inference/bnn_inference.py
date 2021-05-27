# Copyright 2021 Fredrik Wrede, Robin Eriksson, Prashant Singh and Andreas Hellander
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
Sequential Bayesian neural network posterior esitmation
"""
#Imports
from sklearn.model_selection import train_test_split
from sciope.models.bnn_classifier import BNNModel
from sciope.models.bnn_regressor import BNN_regression
from sciope.inference.inference_base import InferenceBase
from sciope.core import core
from sciope.utilities.housekeeping import sciope_logger as ml
from sciope.utilities.priors.gaussian_prior import GaussianPrior
from toolz import partition_all
import pandas as pd
import numpy as np
import dask
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions

def MCinferMOG(data, model, num_monte_carlo, output_dim):
        m = np.empty((num_monte_carlo,1,output_dim))
        S = np.empty((num_monte_carlo,1,output_dim,output_dim))

        # Take samples from posterior predictive, one sample is a Gaussian
        for i in range(num_monte_carlo):
            normal = model.model(data)
            m[i] = normal.mean()
            S[i] = normal.covariance()

        m_hat = m.mean(axis=0) #mean of a mixture of gaussians with equal weight
        S_hat_mean = S.mean(axis=0)

        #not efficent, avoid loop
        # following corresponds to:
        # (m_i - \hat{m})(m_i - \hat{m})^T
        matmul = np.empty((num_monte_carlo, 1, output_dim,output_dim))
        for i in range(num_monte_carlo):
            inner = m[i] - m_hat
            matmul[i] = np.matmul(inner.T, inner)

        #1/N \sum_{i=1}^N S_i + (m_i - \hat{m})(m_i - \hat{m})^T
        S_hat = S_hat_mean + matmul.mean(axis=0) #Covariance of a mixture of gaussians

        return m_hat, S_hat

def _inBin(data, thetaOld, thetaNew):
    flag = None
    for j in range(thetaNew.shape[1]):
        flag_max = (thetaOld[:,j] < max(thetaNew[:,j]))
        flag_min = (thetaOld[:,j] > min(thetaNew[:,j]))
        if j > 0:
            flag = flag & flag_max & flag_min
        else:
            flag = flag_max & flag_min
  
    return data[flag,:,:], thetaOld[flag,:]
class CategoricalSampler():

    def __init__(self, num_bins=5, use_thresh=True, adaptive=False, threshold=0.05):
        self.num_bins = num_bins
        self.probs = None
        self.bins = None
        self.threshold = threshold
        self.use_thresh = use_thresh
        self.adaptive = adaptive
    
    def sample(self, num_samples, chunk_size):

        samples = self._sample_local_adaptive(self.probs,
                                              self.bins,
                                              num_samples,
                                              self.use_thresh,
                                              self.threshold,
                                              chunk_size)
        return samples

    def _sample_local_adaptive(self,probs, bins, num_samples=1000, use_thresh=False, 
                            thresh=0.8, 
                            chunk_size=0):
        probs_pred = tf.reduce_mean(tf.math.log(probs), axis=0).numpy()
        
        if use_thresh:
            if self.adaptive:
                argsort = np.argsort(probs_pred)[0]
                max_remove = int(np.ceil((1-thresh)*probs_pred.shape[1]))
                probs_pred[:,argsort[:max_remove]] = -np.inf
                probs_pred = probs_pred - np.log(np.sum(np.exp(probs_pred)))
            else:
                probs_pred[probs_pred < np.log(thresh)] = -np.inf

        dist = tfd.Categorical(
        logits=probs_pred, probs=None, dtype=tf.int32, validate_args=True,
        allow_nan_stats=False, name='Categorical')

        samples_bins = dist.sample(num_samples)
        
        samples = np.empty((num_samples,2))
        for e,i in enumerate(samples_bins):
            interval = bins[i.numpy()[0]]
            u1 = np.random.uniform(interval[0].left, interval[0].right)
            u2 = np.random.uniform(interval[1].left, interval[1].right)
            samples[e] = np.array([u1,u2])
        if chunk_size > 0:
            samples = partition_all(chunk_size, samples)
            samples = np.array(list(samples))
        return samples

    def _exp_adaptive_thresh(self, start_thresh, growth, rounds):
        adaptive_thresh = start_thresh + (1 - start_thresh)/rounds**growth*np.arange(1,rounds+1)**growth
        return adaptive_thresh

class BNNClassifier():
    """
    """

    def __init__(self, data, sim, prior_function, num_bins=10,
                 num_monte_carlo=500,
                 verbose=False,
                 use_logger=False):

        self.name = 'BNN Classifier'
        #super(BNN, self).__init__(self.name, data, sim, use_logger)

        self.prior_function = prior_function.draw
        self.num_bins = num_bins
        self.num_monte_carlo = num_monte_carlo
        self.use_logger = use_logger #TODO: use super at production ready
        self.sim = sim               #TODO: use super at production ready
        self.data = data             #TODO: use super at production ready
        self.verbose = verbose
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Sequential Bayesian neural network posterior esitmator initialized")

    def _create_train_val(self, num_bins, train_size=0.8, seed=2):
        """
        num_bins = number of bins per parameter, total bins = num_bins^2 
        """
        df = pd.DataFrame(dict(A=self.train_thetas[:,0], B=self.train_thetas[:,1]))
        d1 = df.assign(
            A_cut=pd.cut(df.A, num_bins),
            B_cut=pd.cut(df.B, num_bins)
        )
        d2 = d1.assign(
            A_label=pd.Categorical(d1.A_cut).rename_categories(range(num_bins)),
            B_label=pd.Categorical(d1.B_cut).rename_categories(range(num_bins))
        )
        d3 = d2.assign(cartesian_label=pd.Categorical(d2.filter(regex='_label').apply(tuple, 1)))
        d4 = d3.assign(cartesian_cut=pd.Categorical(d3.filter(regex='_cut').apply(tuple, 1)))
        labels = pd.Categorical(d4.cartesian_label).rename_categories(range(len(pd.Categorical(d4.cartesian_label).categories))).to_numpy()
        all_ = tf.keras.utils.to_categorical(labels)
        d5 = d4.assign(label=labels)
        bins_ = pd.Categorical(d5.cartesian_cut).categories
        
        dummy = list(range(len(self.train_thetas)))

        self.train_ts, self.val_ts, self.train_thetas, self.val_thetas = train_test_split(self.train_ts, self.train_thetas, train_size=train_size, random_state=seed)
        _, _, train_, val_ = train_test_split(dummy, all_, train_size=train_size, random_state=seed)
        
        return train_, val_, bins_


    def infer(self, num_samples, num_rounds,
              chunk_size=10, seed=None):
        np.random.seed(seed)
        theta = []
        local_sampler = CategoricalSampler(num_bins=self.num_bins)

        try:

            graph_dict = core.get_graph_chunked(self.prior_function, self.sim,
                                            batch_size=num_samples,
                                            chunk_size=chunk_size)                          
            
            for i in range(num_rounds):

                samples, data = dask.compute(graph_dict["parameters"], 
                                            graph_dict["trajectories"])
                samples = core._reshape_chunks(samples)
                data = np.array(data)
                if self.verbose:
                    print('data shape: ', data.shape)
                
                #Reshaping for NN
                # standard is num_chunks x chunk_size x ensemble_size x num_species x time_points
                # new shape num_chunks*chunk_size*ensemble_size x time points x num_species
                data = data.reshape((np.prod(data.shape[:3]), 
                                     data.shape[-1], 
                                     data.shape[-2]))
                theta.append(samples)
                if i > 0:
                    data_, samples_ = _inBin(data, samples, theta[i])
                    data = np.append(data, data_, axis=0)
                    samples = np.append(samples, samples_, axis=0)
                
                #TODO: for every 2 combinations in parameter space
                #TODO: Change _create_train_val to not depend on self.train_thetas and
                #      self.train_ts
                self.train_thetas = samples
                self.train_ts = data

                #Create bins from continous data
                train_, val_, bins_ = self._create_train_val(self.num_bins)

                input_shape = (data.shape[-2],data.shape[-1])
                output_shape = len(bins_)
                
                num_train_examples = len(data)
                
                bnn = BNNModel(input_shape, output_shape, num_train_examples)
                if self.verbose:
                    print(bnn.model.summary())
                    print('num bins: ', len(bins_))
                    print('input_shape: ', input_shape)
                    print('data shape: ', data.shape)
                
                bnn.train(self.train_ts, train_, self.val_ts, val_)

                #TODO: adaptive_thresh[i]
                local_sampler.probs = bnn.mc_sampling(self.data, self.num_monte_carlo)
                local_sampler.bins = bins_
                self.prior_function = local_sampler.sample

                graph_dict = core.get_graph_chunked(self.prior_function, self.sim,
                                                    batch_size=num_samples, 
                                                    chunk_size=chunk_size)
        except KeyboardInterrupt:
            return np.array(theta)
        except:
            raise
        return np.array(theta)


class BNNRegressor():

    def __init__(self, data, sim, prior_function,
                 num_monte_carlo=20,
                 verbose=False,
                 use_logger=False):
        self.name = 'BNN Regressor'
        #super(BNN, self).__init__(self.name, data, sim, use_logger)

        self.prior_function = prior_function
        self.num_monte_carlo = num_monte_carlo
        self.use_logger = use_logger #TODO: use super at production ready
        self.sim = sim               #TODO: use super at production ready
        self.data = data             #TODO: use super at production ready
        self.verbose = verbose
        self._train_hyperparams = {'batch_size':256, 'epochs':400, 'verbose':False}
        self._bnn_complied = False
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Sequential Bayesian neural network posterior esitmator initialized")

    def _construct_bnn(self, input_shape, output_shape, num_train_examples, conv_channel=[25, 6], 
                dense_channel=20,
                kernel_size=5,
                pooling_len=10,
                learning_rate=0.001,
                add_normal = True,
                problem_name='noname', 
                use_logger=False):
        
        self.model = BNN_regression(input_shape, output_shape, num_train_examples, conv_channel, 
                dense_channel, 
                kernel_size,
                pooling_len,
                add_normal,
                problem_name, 
                use_logger)

        self.model._compile_model(learning_rate)
        
        self._bnn_complied = self.model._compiled_model
    
    def _train(self, inputs, targets, val_inputs, val_targets):

        batch_size = self._train_hyperparams['batch_size']
        epochs = self._train_hyperparams['epochs']
        verbose = self._train_hyperparams['verbose']

        self.model.train(inputs, targets, val_inputs, val_targets, 
               batch_size=batch_size,
               epochs=epochs, 
               verbose=verbose)


    def _correct_proposal(self):
        """TODO"""

        return 
    
    def infer(self, num_samples, num_rounds, chunk_size=10, exploit=True, seed=None):
        np.random.seed(seed)
        thetas = []
        data_tot = []
        proposal = self.prior_function

        try:
            for i in range(num_rounds):

                graph_dict = core.get_graph_chunked(proposal.draw, self.sim,
                                            batch_size=num_samples,
                                            chunk_size=chunk_size)

                if self.verbose:
                    print(f"starting round {i}")
                
                #Simulate data
                samples, data = dask.compute(graph_dict["parameters"], 
                                            graph_dict["trajectories"])
                samples = core._reshape_chunks(samples)
                data = np.array(data)
                if self.verbose:
                    print('data shape: ', data.shape)
                
                #Reshaping for NN
                # standard is num_chunks x chunk_size x ensemble_size x num_species x time_points
                # new shape num_chunks*chunk_size*ensemble_size x time points x num_species
                data = data.reshape((np.prod(data.shape[:3]), 
                                     data.shape[-1], 
                                     data.shape[-2]))
                
                #append data from each round
                thetas.append(samples)
                #data_tot.append(data)

                #Split training and validation data
                #inputs, val_inputs, targets, val_targets = train_test_split(np.concatenate(data_tot, axis=0),
                #                                                                           np.concatenate(thetas, axis=0), 
                #                                                                           train_size=0.95)
    
                inputs, val_inputs, targets, val_targets = train_test_split(data,
                                                                            samples, 
                                                                            train_size=0.8)
    
                
                #Construct the BNN model
                output_dim = targets.shape[-1]                                              
                if not self._bnn_complied:
                    self._construct_bnn(inputs.shape[1:], output_dim, inputs.shape[0])

                ############## testing retrain re-compiled model
                if i > 0:
                    self.model._compile_model(prior=self.prior_function, proposal=proposal_tf, default=False)

                if self.model.normal:
                
                    #Start training
                    self._train(inputs, targets, val_inputs, val_targets)

                    #Approximate mixure of gaussians as a single gaussian
                    if exploit:
                        proposal_tf = self.model.model(self.data)
                        #proposal_m, proposal_var = MCinferMOG(self.data, self.model, self.num_monte_carlo, output_dim)
                        proposal_m, proposal_var = proposal_tf.mean(), proposal_tf.covariance()
                        proposal = GaussianPrior(proposal_m[0], S=proposal_var[0])
                        

                    #TODO: correction
                else:
                    raise ValueError("Current implementation only support Gaussian proposals, use add_normal = True when constructing BNN")
                
        except KeyboardInterrupt:
            if self.verbose:
                    print(f"Terminating at round {i}")
            return np.array(thetas)
        except:
            raise
        if self.verbose:
                    print(f"Done after {num_rounds} rounds")
        return proposal, np.array(thetas)


                



            

