# Copyright 2017 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
import numpy as np
import gillespy2
from gillespy2 import SSACSolver, GillesPySolver


class MichaelisMenten(gillespy2.Model):
    def __init__(self, parameter_values=None):
        # initialize Model
        gillespy2.Model.__init__(self, name="Michaelis_Menten")

        # parameters
        rate1 = gillespy2.Parameter(name='rate1', expression=0.0017)
        rate2 = gillespy2.Parameter(name='rate2', expression=0.5)
        rate3 = gillespy2.Parameter(name='rate3', expression=0.1)
        self.add_parameter([rate1, rate2, rate3])

        # Species
        A = gillespy2.Species(name='A', initial_value=301)
        B = gillespy2.Species(name='B', initial_value=120)
        C = gillespy2.Species(name='C', initial_value=0)
        D = gillespy2.Species(name='D', initial_value=0)
        self.add_species([A, B, C, D])

        # reactions
        r1 = gillespy2.Reaction(name="r1", reactants={A: 1, B: 1}, products={C: 1},
                                rate=rate1)

        r2 = gillespy2.Reaction(name="r2", reactants={C: 1}, products={A: 1, B: 1},
                                rate=rate2)

        r3 = gillespy2.Reaction(name="r3", reactants={C: 1}, products={B: 1, D: 1},
                                rate=rate3)
        self.add_reaction([r1, r2, r3])
        self.timespan(np.linspace(0, 100, 101))


if __name__ == '__main__':
    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    model = MichaelisMenten()
    csolver = SSACSolver(model)

    # Specify the simulation density and sampling density
    num_trajectories = 1000
    num_timestamps = 150

    # Generate some data for parameter inference
    model.tspan = np.linspace(1, 100, num_timestamps)
    res = model.run(solver=csolver, show_labels=False, number_of_trajectories=num_trajectories)
    S_trajectories = np.array([res[i][:, 1] for i in range(num_trajectories)]).T

    # Write it to file
    np.savetxt("mm_dataset1000_t500.dat", S_trajectories, delimiter=",")


def simulate(param):
    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    model = MichaelisMenten(parameter_values=param)
    csolver = SSACSolver(model)

    # Set up simulation density
    num_trajectories = 1
    simple_trajectories = model.run(solver=csolver, show_labels=False, number_of_trajectories=num_trajectories)

    # extract time values
    time = np.array(simple_trajectories[0][:, 0])

    # extract just the trajectories for S into a numpy array
    S_trajectories = np.array([simple_trajectories[i][:, 1] for i in range(num_trajectories)]).T

    return S_trajectories
