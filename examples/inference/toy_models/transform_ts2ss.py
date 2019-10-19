

import numpy as np
from create_summary_statistics import summarys
import pickle
import time

from load_data import load_spec


num_timestamps=401
endtime=200
modelname = "vilar_ACR_" + str(endtime) + "_" + str(num_timestamps) + '_all_species'
species = [6]

train_thetas, train_ts = load_spec(modelname=modelname, type = "train", species=species)
validation_thetas = pickle.load(open('datasets/' + modelname + '/validation_thetas.p', "rb" ) )
validation_ts = pickle.load(open('datasets/' + modelname + '/validation_ts.p', "rb" ) )[:,:,species]
test_thetas = pickle.load(open('datasets/' + modelname + '/test_thetas.p', "rb" ) )
test_ts = pickle.load(open('datasets/' + modelname + '/test_ts.p', "rb" ) )

print("data loaded")
start = time.time()
val_sum = np.array([summarys(ts) for ts in validation_ts])
end = time.time()
print("validation summarys generated in ", end-start)
pickle.dump( val_sum, open( 'datasets/' + modelname + '/val_sum.p', "wb" ) )

start = time.time()
train_sum = np.array([summarys(ts) for ts in train_ts])
end = time.time()
print("train summarys generated in ", end-start)
pickle.dump( train_sum, open( 'datasets/' + modelname + '/train_sum.p', "wb" ) )

start = time.time()
test_sum = np.array([summarys(ts) for ts in test_ts])
end = time.time()
print("test summarys generated in ", end-start)
pickle.dump( test_sum, open( 'datasets/' + modelname + '/test_sum.p', "wb" ) )

