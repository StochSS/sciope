# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:11:49 2019

@author: ma10s
"""

from sciope.models.model_base import ModelBase
from tensorflow import keras
from sciope.utilities.housekeeping import sciope_logger as ml
import matplotlib.pyplot as plt


# Class definition
class CNNModel(ModelBase):
    """
    We use keras to define CNN and DNN layers to the model
    """
    

    def __init__(self, use_logger=False, input_shape=(499,3), output_shape=15, con_len=3, con_layers=[25, 50], last_pooling=keras.layers.AvgPool1D):
        self.name = 'CNNModel_con_len' + str(con_len) + '_con_layers' + str(con_layers)
        super(CNNModel, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Artificial Neural Network regression model initialized")
        self.model = construct_model(input_shape,output_shape, con_len=con_len, con_layers=con_layers, last_pooling = last_pooling)
        self.save_as = 'saved_models/cnn_light10'
    
    # train the CNN model given the data
    def train(self, inputs, targets,validation_inputs,validation_targets, batch_size, epochs, learning_rate=0.001,
              save_model = True, val_freq=1, early_stopping_patience=5, plot_training_progress=False):

        es = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1,patience=early_stopping_patience)

        if save_model:
            mcp_save = keras.callbacks.ModelCheckpoint(self.save_as+'.hdf5',
                                                       save_best_only=True, 
                                                       monitor='val_loss', 
                                                       mode='min')
        # Using Adam optimizer
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss='mean_squared_error', metrics=['mae'])
        history = self.model.fit(
                inputs, targets, validation_data=(validation_inputs,
                validation_targets), epochs=epochs, batch_size=batch_size, shuffle=True,
                callbacks=[mcp_save, es], validation_freq=val_freq)
        
        #To avoid overfitting load the model with best validation results after 
        #the first training part.        
        if save_model:
            self.model = keras.models.load_model(self.save_as+'.hdf5')

        #TODO: concatenate history1 and history2 to plot all the training 
        #progress       
        if plot_training_progress:
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])

        return history
            
    # Predict
    def predict(self, xt):
        # predict
        return self.model.predict(xt)

    def load_model(self):
        save_as = self.save_as
        self.model = keras.models.load_model(save_as+'.hdf5')
    
def construct_model(input_shape,output_shape, con_len=3, con_layers = [25, 50, 100], last_pooling=keras.layers.AvgPool1D):
    #TODO: add a **kwargs to specify the hyperparameters
    activation = 'relu'
    dense_activation = 'relu'
    padding = 'same'
    poolpadding = 'valid'


    maxpool = con_len
    levels=3
    batch_mom = 0.99
    reg = None
    # pool = keras.layers.AvgPool1D #
    pool = keras.layers.MaxPooling1D
    model = keras.Sequential()
    depth = input_shape[0]
    
       
    #Add levels nr of CNN layers
    model.add(keras.layers.Conv1D(con_layers[0],con_len, strides=1,
                                  padding=padding, activity_regularizer=reg, 
                                  input_shape=input_shape))
    model.add(keras.layers.Activation(activation))
    model.add(keras.layers.Conv1D(con_layers[0],con_len, strides=1,
                                  padding=padding, activity_regularizer=reg))
    model.add(keras.layers.Activation(activation))

    model.add(pool(maxpool,padding=poolpadding))
    if padding == 'valid':
        depth-=(con_len-1)*3
    depth=depth//maxpool
    
    for i in range(1,len(con_layers)):
        model.add(keras.layers.Conv1D(con_layers[i], con_len, strides=1,
                                      padding=padding, 
                                      activity_regularizer=reg))
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.Conv1D(con_layers[i], con_len, strides=1,
                                      padding=padding, 
                                      activity_regularizer=reg))
        model.add(keras.layers.Activation(activation))
        
        if padding == 'valid':
            depth-=(con_len-1)*2
        if i<len(con_layers)-1:
            model.add(pool(maxpool,padding=poolpadding))
            depth=depth//maxpool
        
    #Using Maxpooling to downsample the temporal dimension size to 1.
    # model.add(keras.layers.MaxPooling1D(depth,padding=poolpadding))
    model.add(last_pooling(depth, padding=poolpadding))
    #Reshape previous layer to 1 dimension (feature state).
    model.add(keras.layers.Flatten())
    
    #Add 3 layers of Dense layers with activation function and Batch Norm.
    for i in range(1,3):
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.BatchNormalization(momentum=batch_mom))
        model.add(keras.layers.Activation(dense_activation))
    
    #Add output layer without Activation or Batch Normalization
    model.add(keras.layers.Dense(output_shape))
        

    model.summary()
    return model  
    
    
   