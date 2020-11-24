from sciope.features.feature_extraction import generate_tsfresh_features
from tsfresh.feature_extraction.settings import EfficientFCParameters
import numpy as np
import pytest


def test_generate_tsfresh_features():
    x = np.random.randn(2, 2, 100)
    features = EfficientFCParameters()
    test = generate_tsfresh_features(x, features)
    assert test.shape == (2, 1500)
