from dask.distributed import get_client, futures_of
from dask import delayed
from toolz import partition_all
import numpy as np


def _cluster_mode():
    try:
        get_client()
        return True
    except ValueError:
        return False


def get_futures(lst):
    """ Loop through items in list to keep order of delayed objects
        when transforming to futures. firect call of futures_of does
        not keep the order of the objects

    Parameters
    ----------
    lst : array-like
        array containing delayed objects
    """
    f = []
    for i in lst:
        f.append(futures_of(i)[0])
    return f

@delayed
def delay_func_chunk(func, chunk):
    res = []
    for x in chunk:
        res.append(func(x))
    return res

def get_summaries(data, func, chunk_size):

    # assumed data is large, make chunks
    #assert len(data)/chunk_size > 1.0, "With chunk_size: {0} will only create 1 chunk, choose a sampler chunk_size".format(chunk_size)
    data_chunked = partition_all(chunk_size, data)
    stats_final = [delayed(func)(chunk) for chunk in data_chunked]
    return stats_final


def get_fixed_mean(data, func, chunk_size):
        """
        Computes the mean over summary statistics on fixed data
        
        Parameters
        ----------
        chunk_size : int
            the partition size when splitting the fixed data. For avoiding many individual tasks
            in dask if the data is large 
        
        Returns
        -------
        ndarray
            scaled distance
        """

        # compute summary stats on fixed data
        stats = get_summaries(data, func, chunk_size)

        mean = delayed(np.mean)

        # reducer 1 mean for each batch
        stats_mean = mean(stats, axis=0)

        # reducer 2 mean over batches
        stats_mean = mean(stats_mean, axis=0, keepdims=True)

        return stats_mean

def get_graph_chunked(param_func, sim_func, summaries_func=None,
                      batch_size=10, chunk_size=2):
    """
    Constructs the dask computational graph involving sampling, simulation,
    summary statistics and distances.

    Parameters
    ----------
    param_func : callable
        the parameter sampling function, see sciope.designs, sciope.sampling
        and sciope.utilities.priors
    sim_func : callable
        the simulator function which takes a parameter point as argument
    summaries_func : callable, optional
        the summaries statistics function which takes a simulation result,
        by default None
    dist_func : callable, optional
        the distance function between , by default None
    batch_size : int, optional
        the number of points being sampled in each batch, by default 10
    chunk_size : int, optional
        decription, by default 2
    ensemble_size : int, optional
        [description], by default 1

        Returns
    -------
    dict
        with keys 'parameters', 'trajectories', 'summarystats' and 'distances'
        values being dask delayed objects
    """

    # worflow sampling with batch size = batch_size

    # Draw from the prior/design
    trial_param = param_func(batch_size, chunk_size=chunk_size)

    #params_chunked = partition_all(chunk_size, trial_param)
    params_chunked = trial_param

    # Perform the simulation

    sim_result = [delay_func_chunk(sim_func, chunk) for chunk in params_chunked]

    # Get the statistic(s)

    if summaries_func is not None:
        
        stats_final = [delay_func_chunk(summaries_func, chunk) for chunk in sim_result]

        # Calculate the distance between the dataset and the simulated result
        
    else:
        stats_final = None
        
    return {"parameters": trial_param, "trajectories": sim_result, 
            "summarystats": stats_final}

def get_distance(dist_func, X, chunked=True):

    if chunked:
        sim_dist = [delay_func_chunk(dist_func, chunk) for chunk in X]
        
    else:
        sim_dist = [delayed(dist_func)(x) for x in X]
    
    return sim_dist

def get_prediction(pred_func, X, chunked=True):

    if chunked:
        pred = [delay_func_chunk(pred_func, chunk) for chunk in X]
        
    else:
        pred = [delayed(pred_func)(x) for x in X]
    
    return pred

def _reshape_chunks(data):
    data = np.asarray(data)
    if len(data.shape) > 1:
        data = data.reshape(-1, data.shape[-1])
        return data
    else:
        new = []
        for chunk in data:
            for point in chunk:
                new.append(point)
        new = np.asarray(new) 
        assert len(new.shape) > 1 
        return np.asarray(new)  

def get_graph_unchunked(param_func, sim_func, summaries_func=None, dist_func=None,
                   fixed=None, batch_size=10, ensemble_size=1):
    """
    Constructs the dask computational graph involving sampling, simulation,
    summary statistics and distances.

    Parameters
    ----------
    param_func : callable
        the parameter sampling function, see sciope.designs, sciope.sampling
        and sciope.utilities.priors
    sim_func : callable
        the simulator function which takes a parameter point as argument
    summaries_func : callable, optional
        the summaries statistics function which takes a simulation result,
        by default None
    dist_func : callable, optional
        the distance function between , by default None
    batch_size : int, optional
        the number of points being sampled in each batch, by default 10
    chunk_size : int, optional
        decription, by default 2
    ensemble_size : int, optional
        [description], by default 1

        Returns
    -------
    dict
        with keys 'parameters', 'trajectories', 'summarystats' and 'distances'
        values being dask delayed objects
    """
    if dist_func is not None:
        assert fixed is not None, "If using distance function, parameter 'fixed' can not be None"

    # worflow sampling with batch size = batch_size

    # Draw from the prior/design
    trial_param = param_func(batch_size) * ensemble_size

    # Perform the simulation

    sim_result = [delayed(sim_func)(x) for x in trial_param]

    # Get the statistic(s)

    if summaries_func is not None:
        
        sim_stats = [delayed(summaries_func)(x) for x in sim_result]

        if ensemble_size > 1:
            stats_final = [delayed(np.mean)(sim_stats[i:i + ensemble_size], 
                           axis=0) for i in range(0, len(sim_stats), ensemble_size)]
        else:
            stats_final = sim_stats

        # Calculate the distance between the dataset and the simulated result
        
        if dist_func is not None:
            sim_dist = [delayed(dist_func)(fixed, stats)
                            for stats in stats_final]
        else:
            sim_dist = None
    else:
        stats_final = None
        sim_dist = None

    return {"parameters": trial_param[:batch_size], "trajectories": sim_result, 
            "summarystats": stats_final, "distances": sim_dist}