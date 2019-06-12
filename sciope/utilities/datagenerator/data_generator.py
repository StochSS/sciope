

from sciope.utilities.priors import uniform_prior
from sciope.data import dataset
import numpy as np
import dask
from vilar_class import Vilar


dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]



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
        print("res_param shape: ", np.array(res_param).shape)
        return res_param, res_sim
    

print("start")

prior_function = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
vilar_model = Vilar()
sim = vilar_model.simulate
print("sim tspan: ", vilar_model.model.tspan)
#Generating data
dg = DataGenerator(prior_function=prior_function, sim=sim)
tp, sim_result = dg.gen(batch_size=10)



print("sim result shape: ",np.array(sim_result).shape)




