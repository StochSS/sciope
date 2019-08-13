# from AutoRegressive_model import simulate, prior
from MovingAverage_model import simulate, prior
import pickle
import numpy as np

sim = simulate

true_param = [0.2,-0.13]
data = simulate(true_param)
# modelname='auto_regression2'
modelname='moving_average2'

n=1000000
train_thetas = np.array(prior(n=n))
train_ts = np.expand_dims(np.array([simulate(p,n=100) for p in train_thetas]),2)

validation_thetas = np.array(prior(n=10000))
validation_ts = np.expand_dims(np.array([simulate(p,n=100) for p in validation_thetas]),2)

test_thetas = np.array(prior(n=10000))
test_ts = np.expand_dims(np.array([simulate(p,n=100) for p in validation_thetas]),2)

pickle.dump( true_param, open( 'datasets/' + modelname + '/true_param.p', "wb" ) )
pickle.dump( data, open( 'datasets/' + modelname + '/obs_data.p', "wb" ) )

pickle.dump( train_thetas, open( 'datasets/' + modelname + '/train_thetas.p', "wb" ) )
pickle.dump( train_ts, open( 'datasets/' + modelname + '/train_ts.p', "wb" ) )

pickle.dump( validation_thetas, open( 'datasets/' + modelname + '/validation_thetas.p', "wb" ) )
pickle.dump( validation_ts, open( 'datasets/' + modelname + '/validation_ts.p', "wb" ) )

pickle.dump( test_thetas, open( 'datasets/' + modelname + '/test_thetas.p', "wb" ) )
pickle.dump( test_ts, open( 'datasets/' + modelname + '/test_ts.p', "wb" ) )
