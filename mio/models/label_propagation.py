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
Semi-supervised Label Propagation Surrogate Model
"""

# Imports
from mio.models.model_base import ModelBase
from sklearn.semi_supervised import label_propagation
from scipy.optimize import basinhopping
from scipy.stats.distributions import entropy
import numpy as np



def get_label_entropies(label_distribution):
    return entropy(label_distribution.T)

def get_average_label_entropy(label_distribution):
    return get_label_entropies(label_distribution).sum()/label_distribution.shape[0]



# Class definition
class LPModel(ModelBase):
    """
    We use the sklearn Label Spreading implementation here.
    """

    def __init__(self, kernel='rbf', alpha=0.7, gamma=0.1, learning_rate=1.0):
        self.name = 'LPModel'
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate

    # Tune parameters of the model
    def optimize(self, min_, max_, niter=10):
        start = np.random.uniform(min_, max_)
        minimizer_bounds = [(min_, max_)]

        minimizer_kwargs = dict(method = "L-BFGS-B", bounds = minimizer_bounds)
        res = basinhopping(self.objective, start, minimizer_kwargs=minimizer_kwargs,
                   niter=niter, disp = False)
        return res.x

    def objective(self, x):
        model = label_propagation.LabelSpreading(kernel=self.kernel, alpha=self.alpha, 
                gamma=x)
        model.fit(self.x, self.y)
        label_prob = model.label_distributions_
        return get_average_label_entropy(label_prob) + self.learning_rate*x**2

    # train the label propagation model given the data
    def train(self, inputs, targets, min_=0.01, max_=30):
        # Scale the training data
        self.x = inputs
        self.y = targets

        # Tune gamma in RBF using basinhopping 
        self.gamma = self.optimize(min_=0.01, max_=30)[0]

        # Propogate labels
        self.model = label_propagation.LabelSpreading(kernel=self.kernel, alpha=self.alpha, 
                    gamma=self.gamma)
        self.model.fit(self.x, self.y)

    # Predict
    def predict(self, xp):
        # predict
        yp = self.model.predict(xp)

        return yp
