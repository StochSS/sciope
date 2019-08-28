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
    

    def __init__(self, use_logger=False,input_shape=(499,3),output_shape=15):
        self.name = 'CNNModel'
        super(CNNModel, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Artificial Neural Network regression model initialized")
        self.model = construct_model(input_shape,output_shape)
        self.save_as = 'saved_models/cnn_light10'
    
    # train the CNN model given the data
    def train(self, inputs, targets,validation_inputs,validation_targets, batch_size, epochs,
              save_model = True, plot_training_progress=False):
        if save_model:
            mcp_save = keras.callbacks.ModelCheckpoint(self.save_as+'.hdf5',
                                                       save_best_only=True, 
                                                       monitor='val_loss', 
                                                       mode='min')

        EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                                      mode='auto')
        #train 40 epochs with batch size = 32
        history1 = self.model.fit(
                inputs, targets, validation_data=(validation_inputs,
                validation_targets), epochs=epochs, batch_size=batch_size, shuffle=True,
                callbacks=[mcp_save])#,EarlyStopping])
        
        #To avoid overfitting load the model with best validation results after 
        #the first training part.        
        if save_model:
            self.model = keras.models.load_model(self.save_as+'.hdf5')
        #train 5 epochs with batch size 4096
        # history2 = self.model.fit(
        #         inputs, targets, validation_data = (validation_inputs,
        #         validation_targets), epochs=5,batch_size=4096,shuffle=True,
        #         callbacks=[mcp_save,EarlyStopping])

                
        #TODO: concatenate history1 and history2 to plot all the training 
        #progress       
        if plot_training_progress:
            plt.plot(history1.history['mae'])
            plt.plot(history1.history['val_mae'])
            
    # Predict
    def predict(self, xt):
        # predict
        return self.model.predict(xt)

    def load_model(self):
        save_as = self.save_as
        self.model = keras.models.load_model(save_as+'.hdf5')
    
def construct_model(input_shape,output_shape):
    #TODO: add a **kwargs to specify the hyperparameters
    activation = 'relu'
    dense_activation = 'relu'
    padding = 'same'
    poolpadding = 'valid'
    con_len = 10
    # lay_size = [int(64*1.5**i) for i in range(10)]
    lay_size = [25, 50, 100]

    maxpool = con_len
    levels=3
    batch_mom = 0.99
    reg = None
    # pool = keras.layers.AvgPool1D #
    pool = keras.layers.MaxPooling1D
    model = keras.Sequential()
    depth = input_shape[0]
    
       
    #Add levels nr of CNN layers
    model.add(keras.layers.Conv1D(lay_size[0],con_len, strides=1, 
                                  padding=padding, activity_regularizer=reg, 
                                  input_shape=input_shape))
    model.add(keras.layers.Activation(activation))
    model.add(keras.layers.Conv1D(lay_size[0],con_len, strides=1,
                                  padding=padding, activity_regularizer=reg))
    model.add(keras.layers.Activation(activation))

    model.add(pool(maxpool,padding=poolpadding))
    if padding == 'valid':
        depth-=(con_len-1)*3
    depth=depth//maxpool
    
    for i in range(1,levels):
        model.add(keras.layers.Conv1D(lay_size[i], con_len, strides=1, 
                                      padding=padding, 
                                      activity_regularizer=reg))
        model.add(keras.layers.Activation(activation))
        model.add(keras.layers.Conv1D(lay_size[i], con_len, strides=1, 
                                      padding=padding, 
                                      activity_regularizer=reg))
        model.add(keras.layers.Activation(activation))
        
        if padding == 'valid':
            depth-=(con_len-1)*2
        if i<levels-1:
            model.add(pool(maxpool,padding=poolpadding))
            depth=depth//maxpool
        
    #Using Maxpooling to downsample the temporal dimension size to 1.
    model.add(keras.layers.MaxPooling1D(depth,padding=poolpadding))
    #Reshape previous layer to 1 dimension (feature state).
    model.add(keras.layers.Flatten())
    
    #Add 3 layers of Dense layers with activation function and Batch Norm.
    for i in range(1,3):
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.BatchNormalization(momentum=batch_mom))
        model.add(keras.layers.Activation(dense_activation))
    
    #Add output layer without Activation or Batch Normalization
    model.add(keras.layers.Dense(output_shape))
        
    #Using Adam optimizer with learning rate 0.001 
    model.compile(optimizer=keras.optimizers.Adam(0.001), 
              loss='mean_squared_error',metrics=['mae'])
    model.summary()
    return model  
    
    
   