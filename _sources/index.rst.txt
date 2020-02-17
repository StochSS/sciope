.. sciope documentation master file, created by
   sphinx-quickstart on Thu Apr 11 20:51:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: logo.png
   :alt: sciope logo
   :width: 100%
   :align: center

Welcome to sciope's documentation!
===============================

Scalable inference, optimization and parameter exploration (sciope)
is a Python 3 package for performing machine learning-assisted likelihood-free inference and model
exploration by large-scale parameter sweeps. It has been designed to simplify the data-driven workflows 
so that users quickly can test and develop new machine learning-assisted approches to likelihood-free inference
and model exploration.  

Salient features and contributions of sciope include:

Systems:

    - Parallel implementation of likelihood-free inference via approximate Bayesian computation (ABC).
    - Built-in large-scale summary statistic, or feature extraction.
    - Support for generating statistical designs including random sampling, factorial design and latin hypercube sampling.
    - Support for training fast data-driven surrogate models of computationally expensive simulations, or observed datasets.
    - Sequential space-filling sampling with the *maximin* criterion.
    - Visualization plugin for model exploration in Jupyter notebooks.
    - Parallel backend using `Dask <https://dask.org>`_

Methodology:

    - A novel, scalable reinforcement learning based summary statistic selection framework that allows the user to obtain a ranking of top $m$ summary statistics from a pool of $n$ candidates.
    - A semi-supervised scalable human-in-the-loop model exploration methodology.


Stochastic Gene Regulatory Networks
-----------------------------------
Sciope has been designed for (but is not limited to) Stochastic Gene Regulatory Networks (GRN). 
Sciope have built-in support and wrappers for `Gillespy2 <https://github.com/GillesPy2/GillesPy2>`_ 
and is part of the development of next-generation `StochSS <https://stochss.org>`_.

Likelihood-free inference
-------------------------
In model inference the task is to fit model parameters to observed experimental data. 
A popular approach for parameter inference in systems biology is Approximate Bayesian Computation (ABC). 
ABC inference requires substantial hyperparameter tuning (such as choosing the prior, tuning acceptance 
thresholds and distance metrics). ABC can become prohibitively slow for high-dimensional problems and it 
is of utmost importance to select informative summary statistics.


Model exploration
-----------------
In model parameter space exploration the modeller's objective is to use the simulator to screen for 
different qualitative behaviours displayed by the model under large variations in parameters. 
Model exploration is often the first step in understanding a system, and applies also when no 
experimental data is available.


Scales from laptops to clusters
-------------------------------
The sheer computational cost associated with simulation and feature extraction for complex high-dimensional and 
stochastic models becomes a bottle-neck both for end-users and method developers. For this reason, Sciope is built 
with a `Dask <https://dask.org>`_ backend to support massive parallelism on platforms from laptops to clouds.

Installation
===============================
You can install sciope with  ``pip``, or by installing from source.

Pip
---

This will install both sciope and other dependencies like NumPy, sklearn,
and so on that are necessary::

   pip install sciope

Install from Source
-------------------

To install sciope from source, clone the repository from `github
<https://github.com/sciope/sciope>`_::

    git clone https://github.com/sciope/sciope.git
    cd sciope
    pip install .

Or do a developer install by using the ``-e`` flag::

    pip install -e .



Examples
========

Model Exploration of a Genetic Toggleswitch
------------------------------------------- 
Here we will implement the stochastic genetic toggleswitch model (Gardner et al. Nature 1999) using GillesPy2
and use it to explore the qualitative output by varying the input parameter space. To be able to visualize and
interact with the simulated output we need to use an interacitve backend in jupyter notebooks. 

.. code-block:: python

   %matplotlib notebook
    # Interactive backend required for model exploration
    import gillespy2
    from gillespy2.solvers.numpy import NumPySSASolver 
    import numpy as np

Setting up the model using Gillespy2

.. code-block:: python

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

Use Sciope's Gillespy2 wrapper to extract simulator and parameters

.. code-block:: python

    from sciope.utilities.gillespy2 import wrapper

    settings = {"solver": NumPySSASolver, "number_of_trajectories":10, "show_labels":True}
    simulator = wrapper.get_simulator(gillespy_model=toggle_model, run_settings=settings, species_of_interest=["U", "V"])

    expression_array = wrapper.get_parameter_expression_array(toggle_model)



Use Latin Hypercube design to generate points which will be sampled from during exploration, the points will
be generated using distributed resources if we have a Dask client initialized (in this example just a local cluster).
Generated points will be persited over the worker nodes (i.e no local memory would be used in case of a real cluster).
Random points from the persisted collection can be gathered by calling :code:`lhc.draw(n_samples)`
Here, we will also use TSFRESH minimal feature set as our summary statistics.

.. code-block:: python

    from dask.distributed import Client
    from sciope.designs import latin_hypercube_sampling
    from sciope.utilities.summarystats.auto_tsfresh import SummariesTSFRESH

    c = Client()

    lhc = latin_hypercube_sampling.LatinHypercube(xmin=expression_array, xmax=expression_array*3)
    #generate points that we will randomly sample from during the exploration
    lhc.generate_array(1000)

    #will use default minimal set of features
    summary_stats = SummariesTSFRESH()

Start Model exploration with StochMET

.. code-block:: python

    from sciope.stochmet.stochmet import StochMET
    met = StochMET(simulator, lhc, summary_stats)

Run a parameter sweep of 500 points

.. code-block:: python

    met.compute(n_points=500, chunk_size=10)

Here we will explore parameter points expressed in feature space (summary statistics) using a dimension reduction method. 
The User can interact with points and label points according to different model behavior. 

Note: The explore function make use of interactive tools such as `ipywidgets <https://github.com/jupyter-widgets/ipywidgets>`_,
it is therefore required that you run in a jupyter notebook with an interactive backend (see the first code cell of this example)

.. code-block:: python

    # Here we use UMAP for dimension reduction
    met.explore(dr_method='umap')

Once at least a few points have been assigned a label, sciope has support for semi-supervised learning using label propagation where 
we can infer the labels of unassigned points. This is a great way of filtering the vast amount of data according qualitative behaviour 
and preferences.

.. code-block:: python
    from sciope.models.label_propagation import LPModel
    #here lets use the dimension reduction embedding as input data
    data = met.dr_model.embedding_

    model_lp = LPModel()
    #train using basinhopping
    model_lp.train(data, met.data.user_labels, min_=0.01, max_=10, niter=50)

.. image:: met.gif
   :alt: sciope met gif
   :width: 100%
   :align: center

Parameter Inference of a Genetic Toggleswitch
---------------------------------------------
This example illustrates the workflow for performing ABC parameter inference of the genetic ToggleSwitch model described above.
We start by importing the required modules.

.. code-block:: python

    from sciope.utilities.priors import uniform_prior
    from sciope.inference.abc_inference import ABC
    from sciope.utilities.distancefunctions import naive_squared
    from sklearn.metrics import mean_absolute_error

We define a search space characterised by the prior function as below.
For the purpose of exposition, the prior is defined around the true parameter vector.

.. code-block:: python

    toggle_model = ToggleSwitch()
    true_param = np.array(list(toggle_model.listOfParameters.items()))[:,1]

    # Use true theta as the reference
    bound = []
    for exp in true_param:
        bound.append(float(exp.expression))
    
    # Set the bounds
    bound = np.array(bound)
    dmin = bound * 0.1
    dmax = bound * 2.0

    # Here we use a uniform prior
    uni_prior = uniform_prior.UniformPrior(dmin, dmax)

Next, we generate the observed dataset by simulating the true parameter point.

.. code-block:: python

    # Generate some fixed(observed) data based on default parameters of model 
    fixed_data = toggle_model.run(solver=NumPySSASolver, number_of_trajectories=100, show_labels=False)

    # Reshape data to (n_points,n_species,n_timepoints)
    fixed_data = np.asarray([x.T for x in fixed_data])

    # and remove timepoints array
    fixed_data = fixed_data[:,1:, :]

We are now ready to define the building blocks of the parameter inference pipeline. 
We instantiate the summary statistics to use, the distance function and finally the ABC object.

.. code-block:: python

    # Function to generate summary statistics 
    summ_func = SummariesTSFRESH()

    # Distance
    ns = naive_squared.NaiveSquaredDistance()

    # Setup abc instance
    abc = ABC(fixed_data, sim=simulator, prior_function=uni_prior, summaries_function=summ_func.compute, distance_function=ns)

Initialise and run ABC.

.. code-block:: python

    # First compute the fixed (observed) mean 
    abc.compute_fixed_mean(chunk_size=2)

    # Run in multiprocessing mode
    res = abc.infer(num_samples=100, batch_size=10, chunk_size=2)

Evaluate parameter inference quality.

.. code-block:: python

    mae_inference = mean_absolute_error(true_param, abc.results['inferred_parameters'])
    print('Mean absolute error in parameter inference = {}'.format(mae_inference))

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
