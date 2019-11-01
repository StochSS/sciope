"""
The vilar Model: Approximate Bayesian Computation Test Run
"""

# Imports
from sciope.utilities.priors import uniform_prior
from sciope.inference import abc_inference
from sciope.utilities.summarystats import burstiness as bs
import numpy as np
# import vilar
from vilar_all_species import Vilar_model
import dask
import pickle
import os
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
        res_param, res_sim =  dask.compute(graph_dict["parameters"], graph_dict["trajectories"])
        res_param = np.squeeze(np.array(res_param))
        res_sim = np.squeeze(np.array(res_sim))
        return res_param, res_sim

    def sim_param(self, param):
        return self.sim(param)



num_timestamps=401
endtime=200

Vilar_ = Vilar_model(num_timestamps=num_timestamps, endtime=endtime)
simulate = Vilar_.simulate

modelname = "vilar_allspecies_" + str(endtime) + "_" + str(num_timestamps)


if not os.path.exists('datasets/'+modelname):
    os.mkdir('datasets/'+modelname)

dmin = [0,    100,    0,   20,   10,   1,    1,   0,   0,   0, 0.5,    0,   0,    0,   0]
dmax = [80,   600,    4,   60,   60,   7,   12,   2,   3, 0.7, 2.5,   4,   3,   70,   300]
true_params = [[50.0, 500.0, 0.01, 50.0, 50.0, 5.0, 10.0, 0.5, 1.0, 0.2, 1.0, 1.0, 2.0, 50.0, 100.0]]
# obs_data = np.zeros((20,num_timestamps,9))
# for i in range(20):
#     obs_data[i,:,:] = simulate(np.array(true_params))
# pickle.dump( true_params, open( 'datasets/' + modelname + '/true_param_pack.p', "wb" ) )
# pickle.dump( obs_data, open( 'datasets/' + modelname + '/obs_data_pack.p', "wb" ) )
#

# Set up the prior

prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax)) # .draw

dg = DataGenerator(prior_function=prior, sim=simulate)
print("generating some data")
# nr=0
# while os.path.isfile('datasets/' + modelname + '/train_thetas_'+str(nr)+'.p'):
#     nr += 1
#
# for nr in range(nr,3):
#     train_thetas = np.zeros((0,15))
#     train_ts = np.zeros((0,num_timestamps,9))
#     for i in range(100):
#         param, ts = dg.gen(batch_size=1000)
#         train_thetas = np.concatenate((train_thetas,param),axis=0)
#         # print("train_ts shape: ", train_ts.shape, ", ts shape: ", ts.shape)
#         train_ts = np.concatenate((train_ts,ts),axis=0)
#
#         print("trainig data shape: train_ts: ", train_ts.shape, ", train_thetas: ", train_thetas.shape)
#
#     print("generating trainig data done, shape: train_ts: ", train_ts.shape, ", train_thetas: ", train_thetas.shape)
#
#     pickle.dump( train_thetas, open( 'datasets/' + modelname + '/train_thetas_'+str(nr)+'.p', "wb" ) )
#     pickle.dump( train_ts, open( 'datasets/' + modelname + '/train_ts_'+str(nr)+'.p', "wb" ) )

validation_thetas = np.zeros((0,15))
validation_ts = np.zeros((0,num_timestamps,9))
for i in range(20):
    param, ts = dg.gen(batch_size=1000)
    validation_thetas = np.concatenate((validation_thetas,param),axis=0)
    validation_ts = np.concatenate((validation_ts,ts),axis=0)
    print("validation data shape: train_ts: ", validation_ts.shape, ", train_thetas: ", validation_thetas.shape)

print("generating validation data done, shape: validation_ts: ", validation_ts.shape, ", validation_thetas: ", validation_thetas.shape)

pickle.dump( validation_thetas, open( 'datasets/' + modelname + '/validation_thetas.p', "wb" ) )
pickle.dump( validation_ts, open( 'datasets/' + modelname + '/validation_ts.p', "wb" ) )


# test_thetas = np.zeros((0,15))
# test_ts = np.zeros((0,num_timestamps,9))
# for i in range(100):
#     param, ts = dg.gen(batch_size=1000)
#     test_thetas = np.concatenate((test_thetas,param),axis=0)
#     test_ts = np.concatenate((test_ts,ts),axis=0)
#     print("test data shape: test_ts: ", test_ts.shape, ", test_thetas: ", test_thetas.shape)
#
# print("generating test data done, shape: test_ts: ", test_ts.shape, ", test_thetas: ", test_thetas.shape)
# pickle.dump( test_thetas, open( 'datasets/' + modelname + '/test_thetas.p', "wb" ) )
# pickle.dump( test_ts, open( 'datasets/' + modelname + '/test_ts.p', "wb" ) )

abc_trial_thetas = np.zeros((0,15))
abc_trial_ts = np.zeros((0,num_timestamps,9))
for i in range(100):
    param, ts = dg.gen(batch_size=1000)
    abc_trial_thetas = np.concatenate((abc_trial_thetas,param),axis=0)
    abc_trial_ts = np.concatenate((abc_trial_ts,ts),axis=0)
    if i%10 == 0:
        print("abc_trial data shape: abc_trial_ts: ", abc_trial_ts.shape, ", abc_trial_thetas: ", abc_trial_thetas.shape)

pickle.dump( abc_trial_thetas, open( 'datasets/' + modelname + '/abc_trial_thetas.p', "wb" ) )
pickle.dump( abc_trial_ts, open( 'datasets/' + modelname + '/abc_trial_ts.p', "wb" ) )









