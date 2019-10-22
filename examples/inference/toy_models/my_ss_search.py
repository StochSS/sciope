from create_summary_statistics import summarys


import numpy as np

import pickle
import time
from normalize_data import normalize_data, denormalize_data
from load_data import load_spec
import vilar
from vilar import Vilar_model





num_timestamps=401
endtime=200
modelname = "vilar_ACR_" + str(endtime) + "_" + str(num_timestamps) + '_all_species'
# parameter range
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]

#6=C, 7=A, 8=R
species = [6]




#Load data
train_thetas, train_ts = load_spec(modelname=modelname, type = "train", species=species)
print("load train data done!")

# train_sum = np.array([summarys(train_ts[i],i) for i in range(len(train_ts))])
#
# pickle.dump(train_sum, open('datasets/' + modelname + '/train_sum.p', "wb"))

train_sum = pickle.load(open('datasets/' + modelname + '/train_sum.p', "rb" ) )

obs_data = pickle.load(open('datasets/' + modelname + '/obs_data.p', "rb" ) )

print("obs data shape: ", obs_data.shape)
