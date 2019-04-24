# README #

Scalable inference, optimization and parameter exploration 
is a Python 3 package for performing model-assisted inference and model
exploration by large-scale parameter sweeps.

### What can the sciope toolbox do? ###

* Surrogate Modeling: 
	- train fast metamodels of computationally expensive problems
	- perform surrogate-assisted model reduction for large-scale models/simulators (e.g., biochemical reaction networks)
* Inference: 
	- perform likelihood-free parameter inference using surrogate modeling or Bayesian optimization
	- perform efficient parameter sweeps based on statistical designs and sampling techniques
* Optimization: 
	- optimize a specified objective function or surrogate model using a variety of approaches

* Model exploration: 
	- perform large distributed parameter sweep applications for any black-box model/simulator which output time series data
	- generates time series features/summary statistics on simulation output and visualize parameter points in feature space
	- interactive labeling of paramater points in feature space according to the users preferences over the diversity of model behaviors
	- supports semi-supervised learning and downstream classifiers
	
* Version 0.1

### How do I get set up? ###

* pip install . --process-dependency-links
* Configuration
* Dependencies
	scikit-learn, SciPy, numpy, gpflowopt, ipywidgets, tsfresh, pandas and dask
* How to run tests
	test suite coming up

### Contribution guidelines ###

* Writing tests
	Ongoing
* Code review
	ToDo
* Other guidelines
	ToDo

### Who do I talk to? ###

* Prashant Singh (prashant.singh@it.uu.se)
* Fredrik Wrede (fredrik.wrede@it.uu.se)
* Andreas Hellander (andreas.hellander@it.uu.se)