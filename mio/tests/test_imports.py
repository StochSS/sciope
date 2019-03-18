def test_data():
    from mio.data import dataset

def test_designs():
    from mio.designs import initial_design_base, factorial_design, latin_hypercube_sampling, random_sampling

def test_features():
    from mio.features import feature_extraction

def test_inference():
    from mio.inference import abc_inference, bandits_abc, inference_base

def test_models():
    from mio.models import ann_regressor, gp_regressor, label_propagation, model_base

def test_sampling():
    from mio.sampling import maximin_sampling, sampling_base

def test_visualize():
    from mio.visualize import interactive_scatter

def test_utilities():
    from mio.utilities.distancefunctions import distance_base, euclidean, manhattan, naive_squared
    from mio.utilities.housekeeping import mio_logger, mio_profiler
    from mio.utilities.mab import mab_base, mab_direct, mab_halving, mab_incremental, mab_sar
    from mio.utilities.priors import prior_base, uniform_prior
    from mio.utilities.summarystats import burstiness, global_max, global_min, summary_base, temporal_mean, temporal_variance
    
