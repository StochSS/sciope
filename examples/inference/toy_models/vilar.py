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
Example: The vilar model
"""
# Initialize
import numpy as np
import gillespy2
from gillespy2.solvers.stochkit import StochKitSolver

import os


class Vilar_model:


    def __init__(self,num_timestamps=401, endtime=200):
        self.num_timestamps = num_timestamps
        self.endtime = endtime

    def simulate(self,param):


        num_timestamps = self.num_timestamps
        endtime = self.endtime

        # Load the model definition
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "StochSS_model/vilar_oscillator_AIYDNg/models/data/vilar_oscillator.xml")
        model_doc = gillespy2.StochMLDocument.from_file(config_file)

        # Here, we create the model object.
        model = model_doc.to_model("vilar")


        # Set model parameters
        param = param.ravel()
        temp_param = model.get_parameter('alpha_A')
        temp_param.set_expression(param[0])

        temp_param = model.get_parameter('alpha_a_prime')
        temp_param.set_expression(param[1])

        temp_param = model.get_parameter('alpha_r')
        temp_param.set_expression(param[2])

        temp_param = model.get_parameter('alpha_r_prime')
        temp_param.set_expression(param[3])

        temp_param = model.get_parameter('beta_a')
        temp_param.set_expression(param[4])

        temp_param = model.get_parameter('beta_r')
        temp_param.set_expression(param[5])

        temp_param = model.get_parameter('delta_ma')
        temp_param.set_expression(param[6])

        temp_param = model.get_parameter('delta_mr')
        temp_param.set_expression(param[7])

        temp_param = model.get_parameter('delta_a')
        temp_param.set_expression(param[8])

        temp_param = model.get_parameter('delta_r')
        temp_param.set_expression(param[9])

        temp_param = model.get_parameter('gamma_a')
        temp_param.set_expression(param[10])

        temp_param = model.get_parameter('gamma_r')
        temp_param.set_expression(param[11])

        temp_param = model.get_parameter('gamma_c')
        temp_param.set_expression(param[12])

        temp_param = model.get_parameter('Theta_a')
        temp_param.set_expression(param[13])

        temp_param = model.get_parameter('Theta_r')
        temp_param.set_expression(param[14])

        # Set up simulation density
        num_sim_trajectories = 1
        model.tspan = np.linspace(0, endtime, num_timestamps)
        simple_trajectories = model.run(solver=StochKitSolver, show_labels=True, number_of_trajectories=num_sim_trajectories)

        # extract time values
        #time = np.array(simple_trajectories[0][:, 0])
        species_list = ['C','A','R']
        # extract just the trajectories for Specie A with index 8 into a numpy array
        s_trajectories = np.array([simple_trajectories[i][:, species_list] for i in range(num_sim_trajectories)]).T

        return s_trajectories.T




def get_model():
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "StochSS_model/vilar_oscillator_AIYDNg/models/data/vilar_oscillator.xml")
    model_doc = gillespy2.StochMLDocument.from_file(config_file)
    # Here, we create the model object.
    # We could pass new parameter values to this model here if we wished.
    model = model_doc.to_model("vilar")
    return model

def get_parameter_names():
    model = get_model()
    return [k for k in model.listOfParameters.keys()]