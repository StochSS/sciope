from train_function import train_routine
import numpy as np

num_timestamps=401
endtime=200
modelname = "vilar_ACR_prior6_" + str(endtime) + "_" + str(num_timestamps)
# parameter range
dmin = [0,    100,    0,   20,   10,   1,    1,   0,   0,   0, 0.5,    0,   0,    0,   0]
dmax = [80,   600,    4,   60,   60,   7,   12,   2,   3, 0.7, 2.5,   4,   3,   70,   300]
species = [[2,6], [3,5], [4,5], [5,6], [5,7], [6,7],[6,8],[7,8]]
print("species len: ", len(species))
modelname = "vilar_allspecies_" + str(endtime) + "_" + str(num_timestamps)
models = ['PEN'] #, 'PEN', 'DNN']
# models = ['PEN', 'DNN']
dataname = "new_approach"
pens = np.arange(15)

for i in range(15):
    print("species: ", species[i])
    train_routine(modelname=modelname, dmin=dmin, dmax=dmax, species=species[i],dataname=dataname, step=1,
                  pooling_len=2, dlay=[400,400,400], model='CNN', pen_nr =pens[i], res_folder="pens")


