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
Sequential Bayesian neural network posterior esitmation
"""
#Imports
from sklearn.model_selection import train_test_split
from sciope.models.bnn_classifier import BNNModel
from sciope.inference.inference_base import InferenceBase
from sciope.core import core
from sciope.utilities.housekeeping import sciope_logger as ml
from toolz import partition_all
import pandas as pd
import numpy as np
import dask
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions



def _sample_local_adaptive(probs, bins, num_samples=1000, use_thresh=False, 
                           thresh=0.8, 
                           chunk_size=0):
    probs_pred = tf.reduce_mean(tf.math.log(probs), axis=0).numpy()
    print(probs_pred.shape)
    if use_thresh:
        argsort = np.argsort(probs_pred)[0]
        max_remove = int(np.ceil((1-thresh)*probs_pred.shape[1]))
        probs_pred[:,argsort[:max_remove]] = -np.inf
        probs_pred = probs_pred - np.log(np.sum(np.exp(probs_pred))) 

    dist = tfd.Categorical(
    logits=probs_pred, probs=None, dtype=tf.int32, validate_args=True,
    allow_nan_stats=False, name='Categorical')

    samples_bins = dist.sample(num_samples)
    print(samples_bins.shape)
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

def _exp_adaptive_thresh(start_thresh, growth, rounds):
    adaptive_thresh = start_thresh + (1 - start_thresh)/rounds**growth*np.arange(1,rounds+1)**growth
    return adaptive_thresh

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

class BNN():
    """
    """

    def __init__(self, data, sim, prior_function, num_bins=10,
                 num_monte_carlo=500,
                 use_logger=False):

        self.name = 'BNN'
        #super(BNN, self).__init__(self.name, data, sim, use_logger)

        self.prior_function = prior_function.draw
        self.num_bins = num_bins
        self.num_monte_carlo = num_monte_carlo
        self.use_logger = use_logger #TODO: use super at production ready
        self.sim = sim               #TODO: use super at production ready
        self.data = data             #TODO: use super at production ready
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Sequential Bayesian neural network posterior esitmator initialized")

    def _create_train_val(self, num_bins, train_size=0.8, seed=36):
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

        try:

            graph_dict = core.get_graph_chunked(self.prior_function, self.sim,
                                            batch_size=num_samples,
                                            chunk_size=chunk_size)                          
            
            for i in range(num_rounds):

                samples, data = dask.compute(graph_dict["parameters"], 
                                            graph_dict["trajectories"])
                samples = core._reshape_chunks(samples)
                data = np.array(data)
                print('data shape: ', data.shape)
                #Reshaping for NN
                # standard is num_chunks x chunk_size x ensemble_size x num_species x time_points
                # new shape num_chunks*chunk_size*ensemble_size x time points x num_species
                data = data.reshape((np.prod(data.shape[:3]), 
                                     data.shape[-1], 
                                     data.shape[-2]))
                
                if i > 0:
                    data_, samples_ = _inBin(data, samples, theta[i])
                    data = np.append(data, data_, axis=0)
                    samples = np.append(samples, samples_, axis=0)
                theta.append(samples)
                
                
                #TODO: for every 2 combinations in parameter space
                #TODO: Change _create_train_val to not depend on self.train_thetas and
                #      self.train_ts
                self.train_thetas = samples
                self.train_ts = data
                print('data shape: ', data.shape)

                train_, val_, bins_ = self._create_train_val(self.num_bins)

                self.train_thetas = np.asarray(self.train_thetas)
                self.train_ts = np.asarray(self.train_ts)
                self.val_thetas = np.asarray(self.val_thetas)
                self.val_ts = np.asarray(self.val_ts)

                input_shape = (data.shape[-2],data.shape[-1])
                output_shape = len(bins_)
                num_train_examples = len(data)
                print('input_shape: ', input_shape)
                bnn = BNNModel(input_shape, output_shape, num_train_examples)
                print(bnn.model.summary())
                bnn.train(self.train_ts, train_, self.val_ts, val_)

                probs = bnn.mc_sampling(self.data, self.num_monte_carlo)

                #TODO: adaptive_thresh[i]
                self.prior_function = lambda x,chunk_size: _sample_local_adaptive(probs, 
                                                    bins_, 
                                                    num_samples=x, 
                                                    chunk_size=chunk_size, 
                                                    thresh=0.05)

                graph_dict = core.get_graph_chunked(self.prior_function, self.sim,
                                            num_samples, chunk_size)
        except KeyboardInterrupt:
            return theta
        except:
            raise
        return theta
            



            

