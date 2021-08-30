# # Copyright 2021 Fredrik Wrede, Robin Eriksson, Prashant Singh and Andreas Hellander
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
Bayesian Neural Network (BNN) regression
Ref:

"""

from sciope.models.model_base import ModelBase
from sciope.utilities.housekeeping import sciope_logger as ml
import tensorflow.compat.v2 as tf
#import tensorflow as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

class BNN_regression(ModelBase):
    
    def __init__(self, input_shape, output_shape, num_train_examples, conv_channel=[25, 6], 
                dense_channel=20, 
                kernel_size=5,
                pooling_len=10,
                add_normal = True,
                problem_name='noname', 
                use_logger=False):

        self.name = 'BNNModel'
        #super(BNNModel, self).__init__(self.name, use_logger) #TODO: use at production ready
        self.use_logger = use_logger                         #TODO: use super at production ready
        self.normal = add_normal
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Bayesian Neural Network regressor model initialized")
        self.model = self._construct_model(input_shape, output_shape, num_train_examples,
                                           conv_channel, 
                                           dense_channel, 
                                           kernel_size, 
                                           pooling_len,
                                           add_normal)
        self._compiled_model = False
        self.model.summary()
    

    def _construct_model(self, input_shape, output_shape, NUM_TRAIN_EXAMPLES, conv_channel, 
                        dense_channel, kernel_size, pooling_len, add_normal):
        """BCNN used in manuscript"""
        
        poolpadding = 'valid'
        pool = tf.keras.layers.MaxPooling1D
        
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /
                                  tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
        
        model_in = tf.keras.layers.Input(shape=input_shape)
        conv_1 = tfp.layers.Convolution1DFlipout(conv_channel[0], kernel_size=kernel_size, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)
        x = conv_1(model_in)
        
        x = pool(pooling_len, padding=poolpadding)(x)
        
        conv_1 = tfp.layers.Convolution1DFlipout(conv_channel[1], kernel_size=kernel_size, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)
        x = conv_1(x)
        
        x = pool(pooling_len, padding=poolpadding)(x)
        
        x = tf.keras.layers.Flatten()(x)
        
        
        dense = tfp.layers.DenseFlipout(dense_channel, kernel_divergence_fn=kl_divergence_function,
                                        activation=None)
        
        x = dense(x)

        if add_normal:
        
            dense = tfp.layers.DenseFlipout(output_shape, kernel_divergence_fn=kl_divergence_function,
                                            activation=None)  
            x = dense(x)

            #Adding a multivariate Gaussian
            param_size = tfp.layers.MultivariateNormalTriL.params_size(output_shape)
            dense = tf.keras.layers.Dense(param_size, activation=None)
            x = dense(x)
            normal = tfp.layers.MultivariateNormalTriL(output_shape)
            model_out = normal(x)
        else:
            dense = tfp.layers.DenseFlipout(output_shape, kernel_divergence_fn=kl_divergence_function,
                                            activation=tf.nn.relu)
            model_out = dense(x)

        model = tf.keras.Model(model_in, model_out)
        
        return model
    
    def _compile_model(self, learning_rate=0.001, prior=None, proposal=None, default=True):
        tf.keras.backend.clear_session()
        tf.keras.backend.set_floatx('float32')

        # If one like to use early stopping
        #es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0.1, verbose=1,
        #                                      patience=5)

        # Model compilation
        if default:
            kl = sum(self.model.losses)
            negloglik = lambda y, rv_y: -rv_y.log_prob(y)
        #importance weighted log loss
        else:
            kl = sum(self.model.losses)
            negloglik = lambda y, rv_y: -(prior.pdf(y.numpy())/proposal.prob(y.numpy()))*rv_y.log_prob(y) + kl
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        if self.normal:
            loss = negloglik
        else:
            loss = 'mse'
        self.model.compile(optimizer, loss=loss,
                         experimental_run_tf_function=False,
                         run_eagerly=True) ## need run_eagerly = True to convert tensor to numpy in loss function, this will make training slower
        self._compiled_model = True
    
    def train(self, inputs, targets, val_inputs, val_targets, 
             batch_size=256,
             epochs=400,
             learning_rate=0.001, 
             patience=5, 
             min_delta=0.001, 
             verbose=False):

        if not self._compiled_model:
            self._compile_model(learning_rate)
        
        self._fit_model(inputs, targets, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        validation_data=(val_inputs, val_targets),#callbacks=[es]
                    )

    def _fit_model(self, inputs, targets, batch_size, epochs, verbose,
                   validation_data):
        self.model.fit(inputs, targets, batch_size=batch_size, epochs=epochs, verbose=verbose,
                     validation_freq=10, validation_data=validation_data)
    
    def predict(self, inputs):
        if self.normal:
            return self.model(inputs)
        else:
            return self.model.predict(inputs)
        
        