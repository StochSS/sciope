from train_function import train_routine
import numpy as np

num_timestamps=401
endtime=200
modelname = "vilar_ACR_prior6_" + str(endtime) + "_" + str(num_timestamps)
# parameter range
dmin = [0,    100,    0,   20,   10,   1,    1,   0,   0,   0, 0.5,    0,   0,    0,   0]
dmax = [80,   600,    4,   60,   60,   7,   12,   2,   3, 0.7, 2.5,   4,   3,   70,   300]
species = [[i] for i in range(9)]
print("species len: ", len(species))
dataname = "new_prior6"
models = ['CNN'] #, 'PEN', 'DNN']
# models = ['PEN', 'DNN']

for i in range(9):
    train_routine(modelname=modelname, dmin=dmin, dmax=dmax, species=species[i],dataname=dataname,
                  pooling_len=2, dlay=[400,400,400], model='CNN')


