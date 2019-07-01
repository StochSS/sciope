from __future__ import division
from dask.distributed import Client, get_client
from sciope.utilities.priors import uniform_prior
from sciope.designs.latin_hypercube_sampling import LatinHypercube
from sciope.utilities.summarystats import burstiness as bs
import numpy as np
import sys
from sciope.utilities.distancefunctions import naive_squared as ns
from sciope.utilities.distancefunctions import euclidean as euc
from sciope.features.feature_extraction import generate_tsfresh_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
from sciope.core import core
import pytest
import gillespy2
from gillespy2.solvers.numpy import NumPySSASolver
import os
import dask


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
    
bound = np.array(bound)
dmin = bound * 0.5
dmax = bound * 2.0

uni_prior = uniform_prior.UniformPrior(dmin, dmax)
lhd = LatinHypercube(dmin, dmax)

def simple_sampler_chunked(n, chunk_size=2):
    res = []
    for i in range(int(n/chunk_size)):
        res.append(dask.delayed(np.random.randn(2,2)))
    return res

def simple_sampler_unchunked(n):
    res = []
    for i in range(n):
        res.append(dask.delayed(np.random.randn(2)))
    return res

def simple_sim(x):
    return x + np.random.randn(2)


def simple_summ(x):
    return np.array([np.sum(x), np.min(x)])


try:
    c =get_client()
except:
    c = Client()


def test_simple_unchunked():
    
    graph_dict = core.get_graph_unchunked(param_func=simple_sampler_unchunked, sim_func=simple_sim, 
                                    summaries_func=simple_summ, batch_size=10, ensemble_size=1)

    assert len(graph_dict["parameters"]) == 10, "Core test failed, dimensions mismatch"
    assert len(graph_dict["trajectories"]) == 10, "Core test failed, dimensions mismatch"
    assert len(graph_dict["summarystats"]) == 10, "Core test failed, dimensions mismatch"

    params, sim, summ = dask.compute(graph_dict["parameters"], graph_dict["trajectories"],graph_dict["summarystats"])

    assert len(params) == 10, "Core test failed, dimensions mismatch"
    assert len(sim) == 10, "Core test failed, dimensions mismatch"
    assert len(summ) == 10, "Core test failed, dimensions mismatch"


    graph_dict = core.get_graph_unchunked(param_func=simple_sampler_unchunked, sim_func=simple_sim, 
                                    summaries_func=simple_summ, batch_size=10, ensemble_size=2)

    assert len(graph_dict["parameters"]) == 10, "Core test failed, dimensions mismatch"
    assert len(graph_dict["trajectories"]) == 20, "Core test failed, dimensions mismatch"
    assert len(graph_dict["summarystats"]) == 10, "Core test failed, dimensions mismatch"

    params, sim, summ = dask.compute(graph_dict["parameters"], graph_dict["trajectories"],graph_dict["summarystats"])

    assert len(params) == 10, "Core test failed, dimensions mismatch"
    assert len(sim) == 20, "Core test failed, dimensions mismatch"
    assert len(summ) == 10, "Core test failed, dimensions mismatch"


    graph_dict = core.get_graph_unchunked(param_func=simple_sampler_unchunked, sim_func=simple_sim, 
                                    batch_size=10, ensemble_size=2)

    assert len(graph_dict["parameters"]) == 10, "Core test failed, dimensions mismatch"
    assert len(graph_dict["trajectories"]) == 20, "Core test failed, dimensions mismatch"
    assert graph_dict["summarystats"] is None, "Core test failed, excpected None"

    params, sim = dask.compute(graph_dict["parameters"], graph_dict["trajectories"])

    assert len(params) == 10, "Core test failed, dimensions mismatch"
    assert len(sim) == 20, "Core test failed, dimensions mismatch"


def test_simple_chunked():
    
    graph_dict = core.get_graph_chunked(param_func=simple_sampler_chunked, sim_func=simple_sim, 
                                    summaries_func=simple_summ, batch_size=10, chunk_size=2)

    assert len(graph_dict["parameters"]) == 5, "Core test failed, dimensions mismatch"
    assert len(graph_dict["trajectories"]) == 5, "Core test failed, dimensions mismatch"
    assert len(graph_dict["summarystats"]) == 5, "Core test failed, dimensions mismatch"

    params, sim, summ = dask.compute(graph_dict["parameters"], graph_dict["trajectories"],graph_dict["summarystats"])

    sim = np.asarray(sim)
    summ = np.asarray(summ)
    params = np.asarray(params)

    assert params.shape == (5, 2, 2), "Core test failed, dimensions mismatch"
    assert sim.shape == (5, 2, 2), "Core test failed, dimensions mismatch"
    assert summ.shape == (5, 2, 2), "Core test failed, dimensions mismatch"


    graph_dict = core.get_graph_chunked(param_func=simple_sampler_chunked, sim_func=simple_sim, 
                                    batch_size=10, chunk_size=2)

    assert len(graph_dict["parameters"]) == 5, "Core test failed, dimensions mismatch"
    assert len(graph_dict["trajectories"]) == 5, "Core test failed, dimensions mismatch"
    assert graph_dict["summarystats"] is None, "Core test failed, excpected None"

    params, sim = dask.compute(graph_dict["parameters"], graph_dict["trajectories"])

    sim = np.asarray(sim)
    params = np.asarray(params)

    assert params.shape == (5, 2, 2), "Core test failed, dimensions mismatch"
    assert sim.shape == (5, 2, 2), "Core test failed, dimensions mismatch"

def test_param_sim():
    n_points = 10
    #graph_dict = core.get_dask_graph(
    #    param_func=uni_prior.draw, sim_func=simulator2, batch_size=n_points)
    #assert len(graph_dict["parameters"]
    #           ) == 10, "Core test failed, dimensions mismatch"
    #assert len(graph_dict["trajectories"]
    #           ) == 10, "Core test failed, dimensions mismatch"
    #assert graph_dict["summarystats"] is None, "Core test failed, expected None"
    #assert graph_dict["distances"] is None, "Core test failed, expected None"

    lhd = LatinHypercube(dmin, dmax)
    lhd.generate_array(n_points)
    graph_dict = core.get_graph_chunked(
        param_func=lhd.draw, sim_func=simulator2, batch_size=n_points, chunk_size=2)
    assert len(graph_dict["parameters"]
               ) == 5, "Core test failed, dimensions mismatch"
    assert len(graph_dict["trajectories"]
               ) == 5, "Core test failed, dimensions mismatch"
    assert graph_dict["summarystats"] is None, "Core test failed, expected None"

    params, sim = dask.compute(graph_dict["parameters"], graph_dict["trajectories"])

    sim = np.asarray(sim)
    params = np.asarray(params)

    assert params.shape == (5, 2, 5), "Core test failed, dimensions mismatch"
    assert sim.shape == (5, 2, 1, 2, 101), "Core test failed, dimensions mismatch"

    # all points have been sampled from lhd, default auto_redesign = True

    graph_dict = core.get_graph_chunked(
        param_func=lhd.draw, sim_func=simulator2, batch_size=n_points, chunk_size=2)
    assert len(graph_dict["parameters"]
               ) == 5, "Core test failed, dimensions mismatch"
    assert len(graph_dict["trajectories"]
               ) == 5, "Core test failed, dimensions mismatch"
    assert graph_dict["summarystats"] is None, "Core test failed, expected None"

    params, sim = dask.compute(graph_dict["parameters"], graph_dict["trajectories"])

    sim = np.asarray(sim)
    params = np.asarray(params)

    assert params.shape == (5, 2, 5), "Core test failed, dimensions mismatch"
    assert sim.shape == (5, 2, 1, 2, 101), "Core test failed, dimensions mismatch"


def test_param_sim_summ():
    lhd = LatinHypercube(dmin, dmax)
    n_points = 10
    lhd.generate_array(n_points)
    summ = lambda x: generate_tsfresh_features(x, MinimalFCParameters())
    graph_dict = core.get_graph_chunked(
        param_func=lhd.draw, sim_func=simulator2, summaries_func=summ, batch_size=n_points, chunk_size=2)
    assert len(graph_dict["parameters"]
               ) == 5, "Core test failed, dimensions mismatch"
    assert len(graph_dict["trajectories"]
               ) == 5, "Core test failed, dimensions mismatch"
    assert len(graph_dict["summarystats"]) == 5, "Core test failed, expected None"

    params, sim, summaries = dask.compute(graph_dict["parameters"], graph_dict["trajectories"], graph_dict["summarystats"])

    sim = np.asarray(sim)
    params = np.asarray(params)
    summaries = np.asarray(summaries)

    assert params.shape == (5, 2, 5), "Core test failed, dimensions mismatch"
    assert sim.shape == (5, 2, 1, 2, 101), "Core test failed, dimensions mismatch"
    assert summaries.shape == (5, 2, 1, 16), "Core test failed, dimensions mismatch"


    fixed_data = np.asarray([simulator2(bound) for p in range(10)])
    print(fixed_data.shape)
    fixed_data = fixed_data.reshape(10, 2, 101)
    
    fixed_mean = core.get_fixed_mean(fixed_data, summ, chunk_size=2)
    
    m, = dask.compute(fixed_mean)
    m = np.asarray(m)
    assert  m.shape == (1, 16), "Core test failed, dimensions mismatch"

    dist_class = ns.NaiveSquaredDistance()

    dist_func = lambda x: dist_class.compute(x, m)

    dist = core.get_distance(dist_func, graph_dict["summarystats"])

    assert len(dist) == 5, "Core test failed, dimesnion mismatch"

    dist_res, = dask.compute(dist)
    dist_res = np.asarray(dist_res)

    assert dist_res.shape == (5, 2, 1, 16), "Core test failed, dimension mismatch"

