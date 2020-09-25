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
Simulator wrapper for gillespy2 models

"""

# Imports
import numpy as np


def _set_model_parameters(params, model, parameters_of_interest):
    """ params - array, needs to have the same order as model.listOfParameters """
    for e, pname in enumerate(parameters_of_interest):
        model.get_parameter(pname).set_expression(params[e])
    return model

def _simulator(params, model, kwargs, species_of_interest, parameters_of_interest):
    
    model_update = _set_model_parameters(params, model, parameters_of_interest)

    res = model_update.run(**kwargs)
    
    tot_res = []
    if kwargs["show_labels"]:
        for n in res:
            tot_res.append([n[species] for species in species_of_interest])
        tot_res = np.asarray(tot_res)

    else:
        tot_res = np.asarray([x.T for x in res]) # reshape to (N, S, T)  
        tot_res = tot_res[:,1:, :] # should not contain timepoints
    
    return tot_res

def get_parameter_expression_array(gillespy_model, parameters_of_interest=[]):
    default_param = []
    if not parameters_of_interest:
        default_param = np.array(list(gillespy_model.listOfParameters.items()))[:,1]
    else:
        for param in parameters_of_interest:
            assert param in gillespy_model.listOfParameters.keys()
            default_param.append(gillespy_model.listOfParameters[param])

    as_array = []
    for exp in default_param:
        as_array.append(float(exp.expression))
    
    return np.array(as_array)

def get_simulator(gillespy_model,  run_settings, species_of_interest=[], parameters_of_interest=[]):

    if "show_labels" not in run_settings.keys():
        run_settings["show_labels"] = True
        if not species_of_interest:
            species_of_interest = list(gillespy_model.listOfSpecies.keys())
        else:
            for species in species_of_interest:
                assert species in gillespy_model.listOfSpecies.keys()
    if not parameters_of_interest:
        parameters_of_interest = list(gillespy_model.listOfParameters.keys())
    else:
        for param in parameters_of_interest:
            assert param in gillespy_model.listOfParameters.keys()

    
    return lambda x : _simulator(x, gillespy_model, run_settings, species_of_interest, parameters_of_interest)