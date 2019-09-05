# Copyright 2019 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
Example: The Lotka-Volterra Predator-Prey model
"""
import numpy as np
import gillespy2
from gillespy2.solvers.stochkit import StochKitSolver


class LotkaVolterra(gillespy2.Model):
    def __init__(self, parameter_values=[1.0, 0.005, 0.6]):
        # initialize Model
        gillespy2.Model.__init__(self, name="Lotka-Volterra")

        # parameters
        params = parameter_values
        rate1 = gillespy2.Parameter(name='rate1', expression=params[0])
        rate2 = gillespy2.Parameter(name='rate2', expression=params[1])
        rate3 = gillespy2.Parameter(name='rate3', expression=params[2])
        self.add_parameter([rate1, rate2, rate3])

        # Species
        A = gillespy2.Species(name='A', initial_value=50)
        B = gillespy2.Species(name='B', initial_value=100)
        self.add_species([A, B])

        # reactions
        r1 = gillespy2.Reaction(name="r1", reactants={A: 1}, products={A: 2},
                                rate=rate1)

        r2 = gillespy2.Reaction(name="r2", reactants={A: 1, B: 1}, products={B: 2},
                                rate=rate2)

        r3 = gillespy2.Reaction(name="r3", reactants={B: 1}, products={},
                                rate=rate3)
        self.add_reaction([r1, r2, r3])
        self.timespan(np.linspace(0, 30, 30))


def simulate(param):
    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    model = LotkaVolterra(parameter_values=param)

    # Set up simulation density
    num_trajectories = 1
    sim_results = model.run(solver=StochKitSolver, show_labels=False, number_of_trajectories=num_trajectories)
    tot_res = np.asarray([x.T for x in sim_results])  # reshape to (N, S, T)
    tot_res = tot_res[:, 1:, :]  # should not contain timepoints
    tot_res = tot_res.reshape((1, 2, 30))
    return tot_res
