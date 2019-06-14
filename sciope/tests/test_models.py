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
Test suite for ML models
"""
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sciope.models import label_propagation
from sciope.models import gp_regressor
from sciope.models import ann_regressor
from sciope.models import svm_regressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import numpy as np
import pytest

# Prepare regression dataset
boston = load_boston()
x, y = shuffle(boston.data, boston.target, random_state=13)
x = x.astype(np.float32)
offset = int(x.shape[0] * 0.9)
x_train, y_train = x[:offset], y[:offset]
x_test, y_test = x[offset:], y[offset:]


# Lower dimensional noisy test function for GPR
def f(inp):
    return inp * np.sin(inp)


x_gp = np.linspace(0.1, 9.9, 20)
x_gp = np.atleast_2d(x_gp).T
y_gp = f(x_gp).ravel()
dy = 0.5 + 1.0 * np.random.random(y_gp.shape)
noise = np.random.normal(0, dy)
y_gp += noise
y_gp = y_gp.reshape(y_gp.size, 1)
x_gp_test = np.atleast_2d(np.linspace(0, 10, 1000)).T
y_gp_test = f(x_gp_test)
dy = 0.5 + 1.0 * np.random.random(y_gp_test.shape)
noise = np.random.normal(0, dy)
y_gp_test += noise


@pytest.fixture
def iris_data():
    data = load_iris()
    idx = np.random.randint(0, len(data.target), 75)
    data.new_target = np.copy(data.target)
    data.new_target[idx] = -1  # unlabeled data points
    data.idx = idx
    return data


def test_lp_model(iris_data):
    model = label_propagation.LPModel()
    model.train(iris_data.data, iris_data.new_target)
    print(model.gamma)


def test_ann_model():
    model = ann_regressor.ANNModel()
    model.train(x_gp, y_gp.reshape(y_gp.size, 1))
    mse = mean_squared_error(y_gp_test, model.predict(x_gp_test))
    assert mse < 10, "ANN regressor test fail, test error too high"


def test_svr_model():
    model = svm_regressor.SVRModel()
    model.train(x_train, y_train.reshape(y_train.size, 1))
    mse = mean_squared_error(y_test, model.predict(x_test))
    assert mse < 15, "SVM regressor test fail, test error too high"


def test_gpr_model():
    model = gp_regressor.GPRModel()
    model.train(x_gp, y_gp)
    mse = mean_squared_error(y_gp_test, model.predict(x_gp_test)[0])
    assert mse < 10, "GP regressor test fail, test error too high"


def test_ann_model_with_logging():
    model = ann_regressor.ANNModel(use_logger=True)
    model.train(x_gp, y_gp.reshape(y_gp.size, 1))
    mse = mean_squared_error(y_gp_test, model.predict(x_gp_test))
    assert mse < 10, "ANN regressor test fail, test error too high"


def test_svr_model_with_logging():
    model = svm_regressor.SVRModel(use_logger=True)
    model.train(x_train, y_train.reshape(y_train.size, 1))
    mse = mean_squared_error(y_test, model.predict(x_test))
    assert mse < 15, "SVM regressor test fail, test error too high"


def test_gpr_model_with_logging():
    model = gp_regressor.GPRModel(use_logger=True)
    model.train(x_gp, y_gp)
    mse = mean_squared_error(y_gp_test, model.predict(x_gp_test)[0])
    assert mse < 10, "GP regressor test fail, test error too high"