"""
The vilar Model: Approximate Bayesian Computation Test Run
"""

# Imports
from sciope.utilities.priors import uniform_prior
from sciope.inference import abc_inference
from sciope.utilities.summarystats import burstiness as bs
import numpy as np
import vilar
from vilar import simulate

from sklearn.metrics import mean_absolute_error

modelname = "vilar_A_100_201"

# Set up the prior
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
print("generating trainig data")
n=100000
train_thetas = prior(n)
train_ts =np.array([simulate(p) for p in train_thetas])
print("generating trainig data done, shape: train_ts: ", train_ts.shape, ", train_thetas: ", train_thetas.shape)

validation_thetas = np.array(prior(n=10000))
validation_ts = np.expand_dims(np.array([simulate(p) for p in validation_thetas]),2)

test_thetas = np.array(prior(n=10000))
test_ts = np.expand_dims(np.array([simulate(p) for p in validation_thetas]),2)

abc_trial_thetas = np.array(prior(n=500000))
abc_trial_ts = np.expand_dims(np.array([simulate(p) for p in abc_trial_thetas]),2)

pickle.dump( true_param, open( 'datasets/' + modelname + '/true_param.p', "wb" ) )
pickle.dump( data, open( 'datasets/' + modelname + '/obs_data.p', "wb" ) )

pickle.dump( train_thetas, open( 'datasets/' + modelname + '/train_thetas.p', "wb" ) )
pickle.dump( train_ts, open( 'datasets/' + modelname + '/train_ts.p', "wb" ) )

pickle.dump( validation_thetas, open( 'datasets/' + modelname + '/validation_thetas.p', "wb" ) )
pickle.dump( validation_ts, open( 'datasets/' + modelname + '/validation_ts.p', "wb" ) )

pickle.dump( test_thetas, open( 'datasets/' + modelname + '/test_thetas.p', "wb" ) )
pickle.dump( test_ts, open( 'datasets/' + modelname + '/test_ts.p', "wb" ) )

pickle.dump( abc_trial_thetas, open( 'datasets/' + modelname + '/abc_trial_thetas.p', "wb" ) )
pickle.dump( abc_trial_ts, open( 'datasets/' + modelname + '/abc_trial_ts.p', "wb" ) )
