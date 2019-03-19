# README #

The (M)odeling, (I)inference and (O)ptimization toolbox is a Python 2 package for performing model-assisted inference and optimization.

### What can the MIO toolbox do? ###

* Surrogate Modeling: 
	- train fast metamodels of computationally expensive problems
	- perform surrogate-assisted model reduction for large-scale models/simulators (e.g., biochemical reaction networks)
* Inference: 
	- perform likelihood-free parameter inference using surrogate modeling or Bayesian optimization
	- perform efficient parameter sweeps based on statistical designs and sampling techniques
* Optimization: 
	- optimize a specified objective function or surrogate model using a variety of approaches
	
* Version 0.1

### How do I get set up? ###

* pip install . --process-dependency-links
* Configuration
* Dependencies
	scikit-learn, SciPy, numpy, gpflowopt, ipywidgets, tsfresh, pandas
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