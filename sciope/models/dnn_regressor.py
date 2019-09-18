# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:11:49 2019

@author: ma10s
"""
from tensorflow.python.keras.callbacks import EarlyStopping

from sciope.models.model_base import ModelBase
from tensorflow import keras
from sciope.utilities.housekeeping import sciope_logger as ml
import matplotlib.pyplot as plt
import numpy as np

# Class definition
class ANNModel(ModelBase):
    """
    We use keras to define CNN and DNN layers to the model
    """

    def __init__(self, use_logger=False, input_shape=(499, 3), output_shape=15, layers=[100,100,100], modelname="None"):
        self.name = 'DNNModel_l' + str(layers)
        self.modelname = modelname
        super(ANNModel, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Artificial Neural Network regression model initialized")
        self.model = construct_model(input_shape, output_shape, layers = layers)
        self.save_as = 'saved_models/'  + self.modelname + "_" +  self.name

    # train the ANN model given the data
    def train(self, inputs, targets, validation_inputs, validation_targets, batch_size, epochs, learning_rate = 0.001,
              save_model=True, plot_training_progress=False):
        if save_model:
            mcp_save = keras.callbacks.ModelCheckpoint(self.save_as + '.hdf5',
                                                       save_best_only=True,
                                                       monitor='val_loss',
                                                       mode='min')

        # Using Adam optimizer with learning rate 0.001
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss='mean_squared_error', metrics=['mae'])

        # train 40 epochs with batch size = 32
        history1 = self.model.fit(
            inputs, targets, validation_data=(validation_inputs,
                                              validation_targets), epochs=epochs, batch_size=batch_size, shuffle=True,
            callbacks=[mcp_save])

        # To avoid overfitting load the model with best validation results after
        # the first training part.
        if save_model:
            self.model = keras.models.load_model(self.save_as + '.hdf5')
        # train 5 epochs with batch size 4096
        # history2 = self.model.fit(
        #     inputs, targets, validation_data=(validation_inputs,
        #                                       validation_targets), epochs=20, batch_size=4096, shuffle=True,
        #     callbacks=[mcp_save, EarlyStopping])

        # TODO: concatenate history1 and history2 to plot all the training
        # progress
        if plot_training_progress:
            plt.plot(history1.history['mae'])
            plt.plot(history1.history['val_mae'])

    # Predict
    def predict(self, xt):
        # predict
        return self.model.predict(xt)

    def load_model(self, save_as=None ):
        if not save_as:
            save_as = self.save_as
        self.model = keras.models.load_model(save_as+'.hdf5')


def construct_model(input_shape, output_shape,layers=[100, 100, 100]):
    # TODO: add a **kwargs to specify the hyperparameters
    dense_activation = 'relu'
    batch_mom = 0.99
    model = keras.Sequential()
    new_shape = np.prod(input_shape)
    model.add(keras.layers.Reshape((new_shape,), input_shape=input_shape))

    # Add 3 layers of Dense layers with activation function and Batch Norm.
    for i in range(0, len(layers)):
        if i == 0:
            model.add(keras.layers.Dense(layers[i], input_dim=100))
        else:
            model.add(keras.layers.Dense(layers[i]))

        model.add(keras.layers.BatchNormalization(momentum=batch_mom))
        model.add(keras.layers.Activation(dense_activation))

    # Add output layer without Activation or Batch Normalization
    model.add(keras.layers.Dense(output_shape))


    model.summary()
    return model


