from sklearn.datasets import load_iris
from sciope.models import label_propagation
import numpy as np
import pytest


@pytest.fixture
def iris_data():
    data = load_iris()
    idx = np.random.randint(0, len(data.target), 75)
    data.new_target = np.copy(data.target)
    data.new_target[idx] = -1  # unlabeled data points
    data.idx = idx
    return data


def test_lpmodel(iris_data):
    model = label_propagation.LPModel()
    model.train(iris_data.data, iris_data.new_target)
    print(model.gamma)
