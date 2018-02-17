# Copyright 2017 Prashant Singh, Andreas Hellander and Fredrik Wrede
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example: Michaelis-Menten chemical kinetics
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import gillespy

class MichaelisMenten(gillespy.Model):
    """
        This is an example showcasing a simple Michaelis Menten reaction.
    """

    def __init__(self, parameter_values=[1.0, 110.0, 10.0, 10.0]):

        # Initialize the model.
        gillespy.Model.__init__(self, name="MichaelisMenten")
        #print parameter_values
        #print parameter_values.shape
        
        # Parameters
        k1 = gillespy.Parameter(name='k1', expression=parameter_values[0])
        Km = gillespy.Parameter(name='Km', expression=parameter_values[1])
        mu = gillespy.Parameter(name='mu', expression=parameter_values[2])
        Vmax = gillespy.Parameter(name='Vmax', expression=parameter_values[3])
        
        self.add_parameter([k1,Km,mu,Vmax])
        
        # Species
        S = gillespy.Species(name='S', initial_value=0)
        P = gillespy.Species(name='P', initial_value=0)
        
        self.add_species([S,P])
        
        # Reactions
        rxn1 = gillespy.Reaction(
                name = 'S production',
                reactants = {},
                products = {S:1},
                rate = mu )
                

        rxn2 = gillespy.Reaction(
                name = 'P production',
                reactants = {P:1},
                products = {},
                rate = k1 )

        rxn3 = gillespy.Reaction(
                name = 'S conversion to P',
                reactants = {S:1},
                products = {P:1},
                propensity_function = 'Vmax * S / (Km + S)' )

        self.add_reaction([rxn1,rxn2,rxn3])
        self.timespan(range(150))



if __name__ == '__main__':

    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    simple_model = MichaelisMenten([1.0, 110.0, 10.0, 10.0])
    
    # The model object is simulated with the StochKit solver, and 25 
    # trajectories are returned.
    num_trajectories = 1000
    num_timestamps = 150
    '''
    simple_trajectories = simple_model.run(number_of_trajectories = num_trajectories)
    
    # extract time values
    time = np.array(simple_trajectories[0][:,0]) 

    # extract just the trajectories for S into a numpy array
    S_trajectories = np.array([simple_trajectories[i][:,1] for i in xrange(num_trajectories)]).T
    
    meanTrajs = S_trajectories.mean(1);
    print meanTrajs.item(9)
    '''
    # Generate some data for parameter inference
    simple_model.tspan=range(num_timestamps)
    res = simple_model.run(number_of_trajectories = num_trajectories)
    S_trajectories = np.array([res[i][:,1] for i in xrange(num_trajectories)]).T
    	
    # Write it to file
    np.savetxt("mmDataset1000_t500.dat", S_trajectories, delimiter=",")

    
def simulate(param):
    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    simple_model = MichaelisMenten(parameter_values=param)
    
    # The model object is simulated with the StochKit solver, and 25 
    # trajectories are returned.
    #num_trajectories = 250
    num_trajectories = 1
    simple_trajectories = simple_model.run(number_of_trajectories = num_trajectories)
    
    # extract time values
    time = np.array(simple_trajectories[0][:,0]) 

    # extract just the trajectories for S into a numpy array
    S_trajectories = np.array([simple_trajectories[i][:,1] for i in xrange(num_trajectories)]).T

    return S_trajectories


