

from sciope.utilities.priors import uniform_prior
from sciope.data import dataset
import numpy as np
import dask
import vilar


dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))

n = 100
#batch_size = 100

theta = mm_prior.draw(n=n)


theta.compute()


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
            with keys 'parameters', 'trajectories', 'summarystats' and 'distances'
        """

        # Rejection sampling with batch size = batch_size 

        # Draw from the prior
        trial_param = [self.prior_function.draw() for x in range(batch_size)]

        # Perform the trial
        sim_result = [self.sim(param) for param in trial_param]

        return {"parameters": trial_param, "trajectories": sim_result}
        
    def gen(self, batch_size):

        # Draw from the prior
        # trial_param = [self.prior_function.draw(self.prior_function) for x in range(batch_size)]
        # Perform the trial
        # sim_result = [self.sim(param) for param in trial_param]
        # return [trial_param,sim_result]

        graph_dict = self.get_dask_graph(batch_size=batch_size)
 #       res_param, res_sim = dask.compute(graph_dict["parameters"])#, graph_dict["trajectories"])
        res_param, res_sim = dask.compute(graph_dict["trajectories"])


        return res_param, res_sim
    

print("start")

prior_function = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
sim = vilar.simulate

pri = prior_function.draw()
rr = vilar.simulate(pri)
print("rr: ", rr)

dg = DataGenerator(prior_function=prior_function, sim=sim)
print("type dg: ", type(dg))
[tp, sim_result] = dg.gen(batch_size=10)

#graph_dict = dg.get_dask_graph(batch_size=10)
#res_param, res_sim = dask.compute(graph_dict["parameters"], graph_dict["trajectories"])



print("sim result shape: ",sim_result)




