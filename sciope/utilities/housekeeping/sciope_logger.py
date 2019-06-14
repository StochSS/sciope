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
import tempfile
import os


class Singleton(type):  # pragma: no cover
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SciopeLogger(object, metaclass=Singleton):    # pragma: no cover
    _logger = None

    def __init__(self, log_level=logging.DEBUG):
        self._logger = logging.getLogger("SciopeLogger")
        self._logger.setLevel(log_level)
        log_format = logging.Formatter('\n%(asctime)s \t [%(levelname)-8s | %(filename)-12s:%(lineno)s] : %(message)s')

        now = datetime.datetime.now()
        log_path = os.path.join(tempfile.gettempdir(), "Sciope_logs")
        self._log_dir_path = log_path

        if not os.path.isdir(log_path):
            os.mkdir(log_path)

        filename = logging.FileHandler(log_path + "/log_" + now.strftime("%Y-%m-%d_%H_%M_%S.%f") + ".log")
        stream_handler = logging.StreamHandler()

        filename.setFormatter(log_format)
        stream_handler.setFormatter(log_format)

        self._logger.addHandler(filename)
        self._logger.addHandler(stream_handler)
        self._logger.propagate = False          # To stop duplicate logging

        print("Sciope logger is now ready. Log directory is {}".format(log_path))

    def get_logger(self):
        return self._logger

    def get_log_dir_path(self):
        return self._log_dir_path
