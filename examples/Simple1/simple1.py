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
Example: A simple degradation process of one specie
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import gillespy

class Simple1(gillespy.Model):
    """
    This is a simple example for mass-action degradation of species S.
    """

    def __init__(self, parameter_values=[0.3]):

        # Initialize the model.
        gillespy.Model.__init__(self, name="simple1")
        
        # Parameters
        k1 = gillespy.Parameter(name='k1', expression=parameter_values[0])
        self.add_parameter(k1)
        
        # Species
        S = gillespy.Species(name='S', initial_value=100)
        self.add_species(S)
        
        # Reactions
        rxn1 = gillespy.Reaction(
                name = 'S degradation',
                reactants = {S:1},
                products = {},
                rate = k1 )
        self.add_reaction(rxn1)
        self.timespan(np.linspace(0,20,101))



if __name__ == '__main__':

    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    simple_model = Simple1([0.3])
    
    # The model object is simulated with the StochKit solver, and 25 
    # trajectories are returned.
    num_trajectories = 250
    simple_trajectories = simple_model.run(number_of_trajectories = num_trajectories)
    
    # extract time values
    time = np.array(simple_trajectories[0][:,0]) 

    # extract just the trajectories for S into a numpy array
    S_trajectories = np.array([simple_trajectories[i][:,1] for i in xrange(num_trajectories)]).T
    
    meanTrajs = S_trajectories.mean(1);
    print meanTrajs
    

def simulateTS(param):
    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    simple_model = Simple1(parameter_values=param)
    
    # The model object is simulated with the StochKit solver, and 25 
    # trajectories are returned.
    num_trajectories = 250
    simple_trajectories = simple_model.run(number_of_trajectories = num_trajectories)
    
    # extract just the trajectories for S into a numpy array
    S_trajectories = np.array([simple_trajectories[i][:,1] for i in xrange(num_trajectories)]).T
    
    # Save values to text - 10th value of mean
    meanTrajs = S_trajectories.mean(1);
    
    return meanTrajs

