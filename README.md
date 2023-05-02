![Sciope logo](logo.png)
----------------------------------------------------------


[![Build Status](https://travis-ci.com/StochSS/sciope.svg?branch=master)](https://travis-ci.com/StochSS/sciope)
[![codecov](https://codecov.io/gh/StochSS/sciope/branch/master/graph/badge.svg)](https://codecov.io/gh/StochSS/sciope)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


# README #

Scalable inference, optimization and parameter exploration (sciope)
is a Python 3 package for performing machine learning-assisted inference and model
exploration by large-scale parameter sweeps. Please see the [documentation](https://stochss.github.io/sciope/) for examples.

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
	
* Version 0.4

### How do I get set up? ###

Please see the [documentation](https://stochss.github.io/sciope/) for instructions to install and examples. The easiest way to start using Sciope is through the StochSS online platform (https://app.stochss.org).

### Steps to a successful contribution ###

 1. Fork Sciope (https://help.github.com/articles/fork-a-repo/)
 2. Make the changes to the source code in your fork.
 3. Check your code with PEP8 or pylint. Please limit text to 80 columns wide.
 4. Each feature or bugfix commit should consist of the corresponding code, tests, and documentation.
 5. Create a pull request to the develop branch in Sciope.
 7. Please feel free to use the comments section to communicate with us, and raise issues as appropriate.
 8. The pull request gets accepted and your new feature will soon be integrated into Sciope!

### Who do I talk to? ###

* Prashant Singh (prashant.singh@it.uu.se)
* Fredrik Wrede (fredrik.wrede@it.uu.se)
* Andreas Hellander (andreas.hellander@it.uu.se)

### Citing Sciope ###

To cite Sciope, please reference the [Bioinformatics application note](https://doi.org/10.1093/bioinformatics/btaa673). Sample Bibtex is given below:

```
@article{sciope,
    author = {Singh, Prashant and Wrede, Fredrik and Hellander, Andreas},
    title = "{Scalable machine learning-assisted model exploration and inference using Sciope}",
    journal = {Bioinformatics},
    year = {2020},
    month = {07},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa673},
    url = {https://doi.org/10.1093/bioinformatics/btaa673},
    note = {btaa673},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaa673/33529616/btaa673.pdf},
}

```

