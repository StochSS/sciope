# Copyright 2019 Prashant Singh, Fredrik Wrede and Andreas Hellander
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
Provides very basic profiling
"""

# Imports
import time
from sciope.utilities.housekeeping import sciope_logger as sl


# Very basic function run-time logging
def profile(function_handle, use_profiler=False):   # pragma: no cover
    def wrap(*args, **kwargs):
        if use_profiler:
            logger = sl.SciopeLogger().get_logger()
            start_time = time.time()
            result = function_handle(*args, **kwargs)
            logger.info("Function {0} run time = {1} seconds".format(function_handle, time.time() - start_time))
        return result
    return wrap
