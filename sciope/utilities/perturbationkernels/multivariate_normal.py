# Copyright 2020 Richard Jiang, Prashant Singh, Fredrik Wrede and Andreas Hellander
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
The Multivariate Normal Perturbation Kernel for
Sequential Monte-Carlo Approximate Bayesian Computation (SMC-ABC)
"""

from sciope.utilities.perturbationkernels.kernel_base import PerturbationKernelBase
from scipy.stats import multivariate_normal
import numpy as np


class MultivariateNormalKernel(PerturbationKernelBase):

    def __init__(self, d, cov=None, use_logger=False, adapt=False):

        self.name = 'MultivariateNormalKernel'
        self.use_logger = use_logger
        self.d = d
        self._adapt = adapt
        if cov is not None:
            self.cov = cov
        else:
            self.cov = 0.1 * np.eye(self.d)

        super(MultivariateNormalKernel, self).__init__(self.name, use_logger)

    def pdf(self, x0, x, log=False):
        pdfs = []
        for i in range(np.asarray(x0).shape[0]):
            if log:
                pdfs.append(multivariate_normal.logpdf(x, x0[i], self.cov))
            else:
                pdfs.append(multivariate_normal.pdf(x, x0[i], self.cov))
        return np.vstack(pdfs)

    def rvs(self, x0, num_points=1):
        r = multivariate_normal.rvs(x0, self.cov)
        return r

    def adapt(self, population):
        if self._adapt:
            self.cov = (2.4 / self.d) * np.cov(population, rowvar=False)
