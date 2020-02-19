![Sciope logo](logo.png)
----------------------------------------------------------


[![Build Status](https://travis-ci.com/sciope/sciope.svg?branch=develop)](https://travis-ci.com/sciope/sciope)
[![codecov](https://codecov.io/gh/sciope/sciope/branch/develop/graph/badge.svg)](https://codecov.io/gh/sciope/sciope)


# README #

Scalable inference, optimization and parameter exploration (sciope)
is a Python 3 package for performing machine learning-assisted inference and model
exploration by large-scale parameter sweeps. Please see the [documentation](https://sciope.github.io/sciope/) for examples.

### What can the sciope toolbox do? ###

* Surrogate Modeling: 
	- train fast metamodels of computationally expensive problems
	- perform surrogate-assisted model reduction for large-scale models/simulators (e.g., biochemical reaction networks)
* Inference: 
	- perform likelihood-free parameter inference using parallel ABC
	- train surrogate models (ANNs) as expressive summary statistics for likelihood-free inference
	- perform efficient parameter sweeps based on statistical designs and sampling techniques
* Optimization: 
	- optimize a specified objective function or surrogate model using a variety of approaches

* Model exploration: 
	- perform large distributed parameter sweep applications for any black-box model/simulator which output time series data
	- generates time series features/summary statistics on simulation output and visualize parameter points in feature space
	- interactive labeling of paramater points in feature space according to the users preferences over the diversity of model behaviors
	- supports semi-supervised learning and downstream classifiers
	
* Version 0.2

### How do I get set up? ###

Please see the [documentation](https://sciope.github.io/sciope/) for instructions to install and examples.

### Who do I talk to? ###

* Prashant Singh (prashant.singh@it.uu.se)
* Fredrik Wrede (fredrik.wrede@it.uu.se)
* Andreas Hellander (andreas.hellander@it.uu.se)
