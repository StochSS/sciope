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
Test suit for stochmet (model exploration)

"""

import numpy as np
from sciope.utilities.summarystats import auto_tsfresh
from sciope.utilities.priors import uniform_prior
from sciope.stochmet import stochmet
from gillespy2.solvers.numpy import NumPySSASolver
from sklearn.svm import SVC
from dask.distributed import Client
import gillespy2
import pytest

class ToggleSwitch(gillespy2.Model):
    """ Gardner et al. Nature (1999)
    'Construction of a genetic toggle switch in Escherichia coli'
    """
    def __init__(self, parameter_values=None):
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
        cu = gillespy2.Reaction(name="r1",reactants={}, products={U:1},
                propensity_function="alpha1/(1+pow(V,beta))")
        cv = gillespy2.Reaction(name="r2",reactants={}, products={V:1},
                propensity_function="alpha2/(1+pow(U,gamma))")
        du = gillespy2.Reaction(name="r3",reactants={U:1}, products={},
                rate=mu)
        dv = gillespy2.Reaction(name="r4",reactants={V:1}, products={},
                rate=mu)
        self.add_reaction([cu,cv,du,dv])
        self.timespan(np.linspace(0,50,101))

toggle_model = ToggleSwitch()

# Define simulator function

def set_model_parameters(params, model):
    """ params - array, needs to have the same order as
        model.listOfParameters """
    for e, (pname, p) in enumerate(model.listOfParameters.items()):
        model.get_parameter(pname).set_expression(params[e])
    return model

# Here we use gillespy2 numpy solver, so performance will
# be quite slow for this model

def simulator(params, model):

    model_update = set_model_parameters(params, model)
    num_trajectories = 1  # TODO: howto handle ensembles

    res = model_update.run(solver=NumPySSASolver, show_labels=False,
                           number_of_trajectories=num_trajectories)
    tot_res = np.asarray([x.T for x in res]) # reshape to (N, S, T)  
    tot_res = tot_res[:,1:, :] # should not contain timepoints
    
    return tot_res


def simulator2(x):
    return simulator(x, model=toggle_model)

# Set up the prior

default_param = np.array(list(toggle_model.listOfParameters.items()))[:,1]
bound = []
for exp in default_param:
    bound.append(float(exp.expression))
    
true_params = np.array(bound)
dmin = true_params * 0.5
dmax = true_params * 2.0

uni_prior = uniform_prior.UniformPrior(dmin, dmax)

default_fc_params = {'mean': None,
                         'variance': None,
                         'skewness': None,
                         'agg_autocorrelation':
                         [{'f_agg': 'mean', 'maxlag': 5},
                          {'f_agg': 'median', 'maxlag': 5},
                          {'f_agg': 'var', 'maxlag': 5}]}

summaries = auto_tsfresh.SummariesTSFRESH(features=default_fc_params)



def test_stochmet_toggleswitch_10points():

    #multi-processing mode
    met = stochmet.StochMET(sim=simulator2, sampler=uni_prior, summarystats=summaries)
    met.compute(n_points=10, chunk_size=2)

    np.testing.assert_equal(met.data.s.shape, (10, 1, 12))
    np.testing.assert_equal(met.data.ts.shape, (10, 1, 2, 101))
    np.testing.assert_equal(met.data.x.shape, (10, 5))
    np.testing.assert_equal(met.data.user_labels.shape, (10,))

    #cluster-mode
    c = Client()

    met.compute(n_points=10, chunk_size=2)

    np.testing.assert_equal(met.data.s.shape, (20, 1, 12))
    np.testing.assert_equal(met.data.ts.shape, (20, 1, 2, 101))
    np.testing.assert_equal(met.data.x.shape, (20, 5))
    np.testing.assert_equal(met.data.user_labels.shape, (20,))

    c.close()

def test_stochmet_toggleswitch_100points():

    #multi-processing mode
    met = stochmet.StochMET(sim=simulator2, sampler=uni_prior, summarystats=summaries)
    met.compute(n_points=100, chunk_size=2)

    np.testing.assert_equal(met.data.s.shape, (100, 1, 12))
    np.testing.assert_equal(met.data.ts.shape, (100, 1, 2, 101))
    np.testing.assert_equal(met.data.x.shape, (100, 5))
    np.testing.assert_equal(met.data.user_labels.shape, (100,))

    #cluster-mode
    c = Client()

    met.compute(n_points=100, chunk_size=2)

    np.testing.assert_equal(met.data.s.shape, (200, 1, 12))
    np.testing.assert_equal(met.data.ts.shape, (200, 1, 2, 101))
    np.testing.assert_equal(met.data.x.shape, (200, 5))
    np.testing.assert_equal(met.data.user_labels.shape, (200,))

    c.close()

def test_stochmet_with_prediction():

    uni_prior = uniform_prior.UniformPrior(dmin, true_params*0.6)
    met = stochmet.StochMET(sim=simulator2, sampler=uni_prior, summarystats=summaries)
    met.compute(n_points=50, chunk_size=2)

    x_0 = met.data.s.reshape((50, 12))
    y_0 = np.zeros(50)

    uni_prior = uniform_prior.UniformPrior(true_params*1.5, dmax)
    met = stochmet.StochMET(sim=simulator2, sampler=uni_prior, summarystats=summaries)
    met.compute(n_points=50, chunk_size=2)

    x_1 = met.data.s.reshape((50, 12))
    y_1 = np.ones(50)

    X = np.vstack((x_0, x_1))
    y = np.hstack((y_0,y_1))

    clf = SVC()
    clf.fit(X,y)

    def predictor(x):
        return clf.predict(x)
    
    #multi-processing mode
    uni_prior = uniform_prior.UniformPrior(dmin, dmax)
    met = stochmet.StochMET(sim=simulator2, sampler=uni_prior, summarystats=summaries)
    met.compute(n_points=10, chunk_size=2, predictor=predictor)

    np.testing.assert_equal(met.data.s.shape, (10, 1, 12))
    np.testing.assert_equal(met.data.ts.shape, (10, 1, 2, 101))
    np.testing.assert_equal(met.data.x.shape, (10, 5))
    np.testing.assert_equal(met.data.user_labels.shape, (10,))
    np.testing.assert_equal(met.data.y.shape, (10, 1))


    #cluster-mode
    c = Client()

    met.compute(n_points=10, chunk_size=2, predictor=predictor)

    np.testing.assert_equal(met.data.s.shape, (20, 1, 12))
    np.testing.assert_equal(met.data.ts.shape, (20, 1, 2, 101))
    np.testing.assert_equal(met.data.x.shape, (20, 5))
    np.testing.assert_equal(met.data.user_labels.shape, (20,))
    np.testing.assert_equal(met.data.y.shape, (20, 1))

    c.close()


    










