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
Factorial Initial Design
"""

# Imports
from sciope.designs.initial_design_base import InitialDesignBase
from sciope.utilities.housekeeping import sciope_logger as ml
from toolz import partition_all
from dask import delayed
import dask.array as da
import numpy as np


# Class definition
class FactorialDesign(InitialDesignBase):
    """
    Class definition for Factorial design

    Properties/variables:
    * name						(FactorialDesign)
    * levels					(the number of levels in the factorial design)
    * xmin						(lower bound of multi-dimensional space encompassing generated points)
    * xmax						(upper bound of multi-dimensional space encompassing generated points)
    * outlier_column_indices	(columns containing outliers)
    * logger                    (a logging object to display/save events)
    * use_logger     			(a boolean variable controlling whether logging is enabled or disabled)


    Methods:
    * generate					(returns a delayed object that can generated the desired number of samples)
    """

    def __init__(self, levels, xmin, xmax, use_logger=False):
        """
        Initialize a factorial design with specified parameters
        
        Parameters
        ----------
        levels : integer
            The number of levels of the factorial design. Number of generated points will be levels^dimensionality
        xmin : vector or 1D array
            Specifies the lower bound of the hypercube within which the design is generated
        xmax : vector or 1D array
            Specifies the upper bound of the hypercube within which the design is generated
        use_logger : bool, optional
            controls whether logging is enabled or disabled, by default False
        """
        name = 'FactorialDesign'
        super(FactorialDesign, self).__init__(name, xmin, xmax, use_logger)
        self.levels = levels
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Factorial design in {0} dimensions initialized".format(len(self.xmin)))

    
    def generate(self):
        """
        Sub-classable method for generating a factorial design of specified 'levels' in the given domain.
        The number of generated points is levels^d.
        
        Returns
        -------
        dask.delayed
            
        """
        if hasattr(self, 'random_idx'):
            del self.random_idx

        # Get grid coordinates
        grid_coords = [da.linspace(lb, ub, num=self.levels) for lb, ub in zip(self.xmin, self.xmax)]

        # Generate the full grid
        x = da.meshgrid(*grid_coords)
        dim_idx = [item.ravel() for item in x]
        x = da.vstack(dim_idx).T
        x = x.rechunk(('auto', x.shape[1]))
        if self.use_logger:
            self.logger.info("Factorial design: generated {0} points in {1} dimensions".format(len(x), len(self.xmin)))
        self.generated = x
        return x
    

    def draw(self,n_samples, chunk_size=1, auto_redesign=True):
        """
        Draw specified number of points from a generated LHD

        Parameters
        ----------
        n_samples : integer
            []
        n: integer
            []
        auto_redesign : boolean
            []

        Returns
        -------
        vector/array
        """
        if not hasattr(self, 'generated'):
            self.generate()

        if not hasattr(self, 'random_idx'):
            self.random_idx = np.arange(self.generated.shape[0])
        
        len_random = len(self.random_idx)
        if len_random == 0:
            del self.generated
            del self.random_idx
            if auto_redesign:
                if self.use_logger:
                    self.logger.info("{0} points left to draw form Latin hypercube design:\
                    computing new design for {0} points".format(n))
                return self.draw(n_samples, chunk_size=chunk_size)
            else:
                raise(ValueError)("{0} points left to draw form Latin hypercube design:\
                clearing design".format(len_random))

        if n_samples > len_random: 
            if self.use_logger:
                self.logger.info("Only {0} points left to draw form Latin hypercube design:\
                    setting n_samples to {0}".format(len_random))
            idx = range(len_random)
        else:   
            idx = np.random.choice(range(len_random), n_samples, replace=False)
        
        idx_chunks = partition_all(chunk_size, idx)
        delays = []
        for i in idx_chunks:
            i = list(i)
            rand_idx = self.random_idx[i]
            delay = self.generated[rand_idx].to_delayed()[0]
            delays.append(delay[0])
            
        self.random_idx = np.delete(self.random_idx, idx)    

        return delays

