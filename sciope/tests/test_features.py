from sciope.features.feature_extraction import generate_tsfresh_features
from tsfresh.feature_extraction.settings import EfficientFCParameters
import numpy as np
import pytest


def test_generate_tsfresh_features():
    X = np.random.randn(2, 100)
    features = EfficientFCParameters()
    test = generate_tsfresh_features(X, features)
