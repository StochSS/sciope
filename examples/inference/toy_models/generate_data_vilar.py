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
import dask

from sklearn.metrics import mean_absolute_error


class DataGenerator:

    def __init__(self, prior_function, sim):
        self.prior_function = prior_function
        self.sim = dask.delayed(sim)

    def get_dask_graph(self, batch_size):
        """
        Constructs the dask computational graph invloving sampling, simulation, summary statistics
        and distances.

        Parameters
        ----------
        batch_size : int
            The number of points being sampled in each batch.

        Returns
        -------
        dict
            with keys 'parameters', 'trajectories'
        """

        # Rejection sampling with batch size = batch_size

        # Draw from the prior
        trial_param = [self.prior_function.draw() for x in range(batch_size)]

        # Perform the trial
        sim_result = [self.sim(param) for param in trial_param]

        return {"parameters": trial_param, "trajectories": sim_result}

    def gen(self, batch_size):
        graph_dict = self.get_dask_graph(batch_size=batch_size)
        res_param, res_sim = dask.compute(graph_dict["parameters"], graph_dict["trajectories"])
        return np.array(res_param), np.array(res_sim)

    def sim_param(self, param):
        return self.sim(param)



modelname = "vilar_A_100_201"


true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
data = 5

# Set up the prior
dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax)) # .draw

dg = DataGenerator(prior_function=prior, sim=simulate)
print("generating some data")

tp, sim_result = dg.gen(batch_size=100)
print("generating some data done: shape: ", tp.shape, sim_result.shape)


print("generating trainig data")
n=100000
train_thetas = np.squeeze( np.array(dask.compute( prior(n))))
train_ts =np.array([simulate(p) for p in train_thetas])
print("generating trainig data done, shape: train_ts: ", train_ts.shape, ", train_thetas: ", train_thetas.shape)

validation_thetas = np.squeeze( np.array(dask.compute(np.array(prior(n=10000)))))
validation_ts = np.expand_dims(np.array([simulate(p) for p in validation_thetas]),2)

test_thetas = np.squeeze( np.array(dask.compute(np.array(prior(n=10000)))))
test_ts = np.expand_dims(np.array([simulate(p) for p in validation_thetas]),2)

abc_trial_thetas = np.squeeze( np.array(dask.compute( np.array(prior(n=500000)))))
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
