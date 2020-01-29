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
Example: The toggle switch model
"""
# Initialize
import numpy as np
from gillespy2.solvers.cpp import SSACSolver
from sciope.utilities.summarystats import auto_tsfresh
from sciope.utilities.priors import uniform_prior
from sciope.inference.abc_inference import ABC
from sklearn.metrics import mean_absolute_error
from sciope.utilities.distancefunctions import naive_squared
import gillespy2


class ToggleSwitch(gillespy2.Model):
    """ Gardner et al. Nature (1999)
    'Construction of a genetic toggle switch in Escherichia coli'
    """

    def __init__(self):
        # Initialize the model.
        gillespy2.Model.__init__(self, name="toggle_switch")
        # Parameters
        alpha1 = gillespy2.Parameter(name='alpha1', expression=1)
        alpha2 = gillespy2.Parameter(name='alpha2', expression=1)
        beta = gillespy2.Parameter(name='beta', expression="2.0")
        gamma = gillespy2.Parameter(name='gamma', expression="2.0")
        mu = gillespy2.Parameter(name='mu', expression=1.0)
        self.add_parameter([alpha1, alpha2, beta, gamma, mu])

        # Species
        U = gillespy2.Species(name='U', initial_value=10)
        V = gillespy2.Species(name='V', initial_value=10)
        self.add_species([U, V])

        # Reactions
        cu = gillespy2.Reaction(name="r1", reactants={}, products={U: 1},
                                propensity_function="alpha1/(1+pow(V,beta))")
        cv = gillespy2.Reaction(name="r2", reactants={}, products={V: 1},
                                propensity_function="alpha2/(1+pow(U,gamma))")
        du = gillespy2.Reaction(name="r3", reactants={U: 1}, products={},
                                rate=mu)
        dv = gillespy2.Reaction(name="r4", reactants={V: 1}, products={},
                                rate=mu)
        self.add_reaction([cu, cv, du, dv])
        self.timespan(np.linspace(0, 50, 101))


def set_model_parameters(params, model):
    """ params - array, needs to have the same order as
        model.listOfParameters """
    for e, (pname, p) in enumerate(model.listOfParameters.items()):
        model.get_parameter(pname).set_expression(params[e])
    return model


def get_true_param():
    """
    Return the 'true' parameter values to be inferred in a test run
    :return: the 'true' or reference parameter set
    """
    model = ToggleSwitch()
    default_param = np.array(list(model.listOfParameters.items()))[:, 1]  # take default from model as reference
    true_param = []
    for exp in default_param:
        true_param.append(float(exp.expression))

    # set the bounds
    true_param = np.array(true_param)
    return true_param


def get_bounds():
    """
    return the default bounds
    :return: bounds in each dimension as a list
    """
    fixed_point = get_true_param()
    dmin = fixed_point * 0.1
    dmax = fixed_point * 2.0
    return dmin, dmax


def get_fixed_data():
    """
    generate fixed data for inference
    :return: a dataset of 100 trajectories
    """
    model = ToggleSwitch()
    fixed_data = model.run(solver=SSACSolver, number_of_trajectories=100, show_labels=False)

    # reshape data to (n_points,n_species,n_timepoints)
    fixed_data = np.asarray([x.T for x in fixed_data])

    # and remove timepoints array
    fixed_data = fixed_data[:, 1:, :]
    return fixed_data


def simulate(params):
    """
    Instantiate a model and simulate input parameters
    :param params: input parameters
    :return:
    """
    model = ToggleSwitch()
    model_update = set_model_parameters(params, model)
    num_trajectories = 1  # TODO: howto handle ensembles

    res = model_update.run(solver=SSACSolver, show_labels=False,
                           number_of_trajectories=num_trajectories)
    tot_res = np.asarray([x.T for x in res])  # reshape to (N, S, T)
    tot_res = tot_res[:, 1:, :]  # should not contain timepoints

    return tot_res


def abc_test_run():
    """
    Perform a test abc run
    :return: ABC mean absolute error
    """
    dmin, dmax = get_bounds()
    uni_prior = uniform_prior.UniformPrior(dmin, dmax)
    fixed_data = get_fixed_data()
    summ_func = auto_tsfresh.SummariesTSFRESH()
    ns = naive_squared.NaiveSquaredDistance()
    abc = ABC(fixed_data, sim=simulate, prior_function=uni_prior,
              summaries_function=summ_func.compute, distance_function=ns, use_logger=True)
    abc.compute_fixed_mean(chunk_size=2)
    res = abc.infer(num_samples=100, batch_size=10, chunk_size=2)
    true_params = get_true_param()
    mae_inference = mean_absolute_error(true_params, abc.results['inferred_parameters'])
    return mae_inference
