

from sciope.utilities.priors import uniform_prior
from sciope.data import dataset
import numpy as np
import dask
import  vilar


dmin = [30, 200, 0, 30, 30, 1, 1, 0, 0, 0, 0.5, 0.5, 1, 30, 80]
dmax = [70, 600, 1, 70, 70, 10, 12, 1, 2, 0.5, 1.5, 1.5, 3, 70, 120]
mm_prior = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))

n = 100
#batch_size = 100

theta = mm_prior.draw(mm_prior,n=n)


theta.compute()


class DataGenerator():

    def __init__(self,prior_function, sim):
        self.prior_function = prior_function
        self.sim = dask.delayed(sim)
        
    def gen(self,batch_size):

        # Draw from the prior
        trial_param = [self.prior_function.draw(self.prior_function) for x in range(batch_size)]        
        # Perform the trial
        sim_result = [self.sim(param) for param in trial_param]        
        return [trial_param,sim_result]
    
    
print("start")

prior_function = uniform_prior.UniformPrior(np.asarray(dmin), np.asarray(dmax))
sim = vilar.simulate
dg = DataGenerator(prior_function = prior_function, sim = sim)
print("type dg: ", type(dg))
[tp,sim_result] = dg.gen(batch_size=10)
print([s.compute() for s in sim_result])







