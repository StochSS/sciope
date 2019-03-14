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
Data Logging Class
"""

# Imports
import logging
import datetime
import os
from pathlib import Path


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MIOLogger(object, metaclass=Singleton):
    _logger = None

    def __init__(self):
        self._logger = logging.getLogger("MIOLogger")
        self._logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter('\n%(asctime)s \t [%(levelname)-8s | %(filename)-12s:%(lineno)s] : %(message)s')

        now = datetime.datetime.now()
        mio_root_path = Path(__file__).parents[3]
        log_path = mio_root_path/"log"
        dirname = str(log_path.resolve())

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        filename = logging.FileHandler(dirname + "/log_" + now.strftime("%Y-%m-%d_%H:%M:%S.%f") + ".log")
        stream_handler = logging.StreamHandler()

        filename.setFormatter(log_format)
        stream_handler.setFormatter(log_format)

        self._logger.addHandler(filename)
        self._logger.addHandler(stream_handler)

        print("MIO Logger is now ready.")

    def get_logger(self):
        return self._logger
