import pytest
import numpy as np

from topshap.shapley import shapley_bf


def test_shapley_bf():
    D = [
        (np.array([0.5]), 1),
        (np.array([1.0]), 0), 
        (np.array([2.0]), 1)
    ]
    z_test = (np.array([0.0]), 1)

    shapley_values = shapley_bf(D, z_test, K=2, sigma=1)
    answer = [0.8374, -0.0451,  0.0902]

    assert np.allclose(shapley_values, answer, atol=1e-03)

    # Test multiple test points
    Z_test = [
        (np.array([0.0]), 1),
        (np.array([0.0]), 1),
    ]
    shapley_values = shapley_bf(D, Z_test, K=2, sigma=1)
    answer = [0.8374, -0.0451,  0.0902]

    assert np.allclose(shapley_values, answer, atol=1e-03)
