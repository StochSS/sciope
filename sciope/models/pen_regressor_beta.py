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
class PEN_CNNModel(ModelBase):
    """
    We use keras to define CNN and DNN layers to the model
    """
    

    def __init__(self, use_logger=False, input_shape=(499,3), output_shape=15, pen_nr=3):
        self.name = 'PEN_NNModel' + str(pen_nr)
        super(PEN_CNNModel, self).__init__(self.name, use_logger)
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Artificial Neural Network regression model initialized")
        self.model = construct_model(input_shape,output_shape,pen_nr)
        self.pen_nr = pen_nr
        self.save_as = 'saved_models/pen'+str(self.pen_nr)


    # train the CNN model given the data
    def train(self, inputs, targets,validation_inputs,validation_targets,
             save_model = True, plot_training_progress=False):
        self.save_as = 'saved_models/pen'+str(self.pen_nr)

        if save_model:
            mcp_save = keras.callbacks.ModelCheckpoint(self.save_as+'.hdf5',
                                                       save_best_only=True, 
                                                       monitor='val_loss', 
                                                       mode='min')

        # EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
        #                                               mode='auto',
        #                                               baseline=None)
        #train 40 epochs with batch size = 32
        history1 = self.model.fit(
                inputs, targets, validation_data = (validation_inputs,
                validation_targets), epochs=10,batch_size=200,shuffle=True,
                callbacks=[mcp_save])
        
        #To avoid overfitting load the model with best validation results after 
        #the first training part.        
        if save_model:
            self.model = keras.models.load_model(self.save_as+'.hdf5')
        #train 5 epochs with batch size 4096
        # history2 = self.model.fit(
        #         inputs, targets, validation_data = (validation_inputs,
        #         validation_targets), epochs=5,batch_size=4096,shuffle=True,
        #         callbacks=[mcp_save])

                
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
        self.model = keras.models.load_model(self.save_as+'.hdf5')
    
def construct_model(input_shape, output_shape, pen_nr = 3):
    #TODO: add a **kwargs to specify the hyperparameters

    activation = 'relu'
    dense_activation = 'relu'
    padding = 'same'
    poolpadding = 'valid'
    con_len = 1
    lay_size = [100, 50, 10, 50, 50, 20]
    maxpool = con_len
    levels=3
    batch_mom = 0.99
    reg = None
    pool = keras.layers.MaxPooling1D
    model = keras.Sequential()
    depth = input_shape[0]
    
    Input = keras.Input(shape=input_shape)
    #Add levels nr of CNN layers
    layer = keras.layers.Conv1D(lay_size[0],pen_nr, strides=1,
                                  padding='valid', activity_regularizer=reg,
                                  input_shape=input_shape)(Input)
    layer = keras.layers.Activation(activation)(layer)


    
    for i in range(1,levels):
        layer = keras.layers.Conv1D(lay_size[i], con_len, strides=1,
                                      padding=padding, 
                                      activity_regularizer=reg)(layer)
        layer = keras.layers.Activation(activation)(layer)

    poolsize = input_shape[0] - (pen_nr - 1)
    print("poolsize: ", poolsize)
    #Using Maxpooling to downsample the temporal dimension size to 1.
    layer = keras.layers.AvgPool1D(poolsize,padding=poolpadding)(layer)

    # model.add(keras.backend.sum(
    #     x,
    #     axis=1,
    #     keepdims=False
    # ))
    #Reshape previous layer to 1 dimension (feature state).
    layer = keras.layers.Flatten()(layer)
    cut_Input = keras.layers.Lambda(lambda x: x[0:pen_nr,:], )(Input)

    # cut_Input_1d = keras.layers.Lambda(lambda x: keras.backend.reshape(pen_nr*input_shape[1],))(cut_Input)

    # cut_Input_1d = keras.backend.reshape(cut_Input,(-1,pen_nr*input_shape[1],))
    cut_Input_1d = keras.layers.Lambda(lambda x: keras.backend.reshape(x,(-1,pen_nr*input_shape[1],)))(cut_Input)


    layer = keras.layers.concatenate([layer, cut_Input_1d])

    #Add 3 layers of Dense layers with activation function and Batch Norm.
    for i in range(3,6):
        layer = keras.layers.Dense(lay_size[i])(layer)
        # model.add(keras.layers.BatchNormalization(momentum=batch_mom))
        layer = keras.layers.Activation(dense_activation)(layer)
    
    #Add output layer without Activation or Batch Normalization
    layer = keras.layers.Dense(output_shape)(layer)

    model = keras.models.Model(inputs=Input, outputs=layer)

        
    #Using Adam optimizer with learning rate 0.001 
    model.compile(optimizer=keras.optimizers.Adam(0.001), 
              loss='mean_squared_error',metrics=['mae'])
    model.summary()
    return model  
    
    
   