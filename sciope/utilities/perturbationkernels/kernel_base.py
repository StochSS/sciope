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
The Perturbation Kernel Base Class for
Sequential Monte-Carlo Approximate Bayesian Computation (SMC-ABC)
"""

# Imports
from abc import ABCMeta, abstractmethod


class PerturbationKernelBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, use_logger=False):
        self.name = name
        self.use_logger = use_logger

    def adapt(self, samples):
        pass

    @abstractmethod
    def pdf(self, x0, x, log=False):
        pass

    @abstractmethod
    def rvs(self, x0, num_points=1):
        pass
