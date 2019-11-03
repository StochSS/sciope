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
modelname = "vilar_allspecies_" + str(endtime) + "_" + str(num_timestamps)
models = ['CNN'] #, 'PEN', 'DNN']
# models = ['PEN', 'DNN']
dataname = "new_approach"
training_size = [200000, 100000, 30000]
steps = [2,4,8,16]
for i in range(4):
    train_routine(modelname=modelname, dmin=dmin, dmax=dmax, species=species[6], training_size=300000, step=steps[i],
                  dataname=dataname, pooling_len=2, dlay=[400,400,400], model='CNN', res_folder="diff_step_sizes")


