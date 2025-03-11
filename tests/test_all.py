import pytest
import numpy as np
from functools import partial

from topshap.helper import distance, kernel_value
from topshap.topt import BallExpander, Point, shapley_top, kcenter, kcenter_naive, shapley_tknn
from topshap.naive import shapley_bf


def test_kcenter():
    # Basic case with 5 points on a line
    Z_test = [
        (np.array([-2.0]), 1),
        (np.array([-1.0]), 1),
        (np.array([0.0]), 1),
        (np.array([1.0]), 1),
        (np.array([2.0]), 1),
    ]
    clusters, testidx2center = kcenter(Z_test, n_clst=3)
    assert list(clusters.keys()) == [0, 4, 2]

    clusters, testidx2center = kcenter_naive(Z_test, n_clst=3)
    assert list(clusters.keys()) == [0, 4, 2]


def test_build_ball():
    pt_test = Point(x=np.array([0.5]), y=1, idx=0, is_test=True)
    
    # Create sorted augmented list with test point first
    sorted_aug = [
        Point(x=np.array([0.0]), y=1, idx=1, is_test=False),
        Point(x=np.array([0.5]), y=1, idx=0, is_test=True), # test point
        Point(x=np.array([1.5]), y=0, idx=2, is_test=False),
        Point(x=np.array([2.0]), y=1, idx=3, is_test=False),
        Point(x=np.array([2.5]), y=1, idx=4, is_test=True)
    ]
    landmark = np.array([0.0])
    
    expander = BallExpander([], [], kernel_fn=partial(kernel_value, sigma=1))
    expander.testidx2aug = {0: sorted_aug}
    expander.testidx2augidx = {0: 1}
    points, dist_radius = expander.build_ball(pt_test, i=1, landmark=landmark)
    
    # Verify new points added
    assert len(points) == 2
    new_ids = [p.idx for p in points]
    assert 1 in new_ids
    assert 2 in new_ids
    
    # Verify radius calculation
    assert np.isclose(dist_radius, 1)
    
    # Increase radius i
    points, dist_radius = expander.build_ball(pt_test, i=2, landmark=landmark)
    assert len(points) == 3
    assert 3 in [p.idx for p in points]
    assert np.isclose(dist_radius, 1.5)

    # Increase radius i
    points, dist_radius = expander.build_ball(pt_test, i=3, landmark=landmark)
    assert len(points) == 3
    assert np.isclose(dist_radius, float('inf'))

    # Another case with more points before test point
    sorted_aug = sorted_aug[::-1] # reverse the order of sorted_aug
    landmark = np.array([2.5])
    expander.testidx2aug = {0: sorted_aug}
    expander.testidx2augidx = {0: 3}

    new_points, dist_radius = expander.build_ball(pt_test, i=2, landmark=landmark)
    assert len(new_points) == 3
    assert np.isclose(dist_radius, 1.5)


def test_shapley_tknn():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([10.5]), 1),
        (np.array([12.0]), 1),
        (np.array([11.0]), 0),
        (np.array([1.0]), 0)
    ]
    Z_test = [(np.array([0.0]), 1), (np.array([10.0]), 1)]

    shapley_values = shapley_tknn(D, Z_test, K=2, radius=3, kernel_fn=partial(kernel_value, sigma=1), n_clst=2)
    answer = np.array([0.8374, 0.0902, 0.8374, 0.0902, -0.0451, -0.0451]) / 2

    assert np.allclose(shapley_values, answer, atol=1e-03)


def test_shapley_top():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    Z_test = [(np.array([0.0]), 1)]

    top_idx = shapley_top(D, Z_test, kernel_fn=partial(kernel_value, sigma=1), t=1, K=2, n_clst=1)
    assert top_idx == [0]

    top_idx = shapley_top(D, Z_test, kernel_fn=partial(kernel_value, sigma=1), t=2, K=2, n_clst=1)
    assert np.all(top_idx == [0, 1])

    top_idx = shapley_top(D, Z_test, kernel_fn=partial(kernel_value, sigma=1), t=3, K=2, n_clst=1)
    assert top_idx == None # fail to find [0, 1, 2]

    top_idx = shapley_top(D, Z_test, kernel_fn=partial(kernel_value, sigma=1), t=3, K=2, n_clst=1, t_ub=3)
    assert np.all(top_idx == [0])


def test_shapley_top_2dplanes():
    # Test on 2dplanes.arff
    np.random.seed(42)
    
    data = np.genfromtxt('./datasets/2dplanes.arff', delimiter=',', dtype=str, skip_header=15)
    labels = np.where(data[:, -1] == 'P', 1, 0)
    features = data[:, :-1].astype(float)
    data = np.column_stack((features, labels))

    # shuffle the data
    np.random.shuffle(data)

    # split the data into D and Z_test
    ratio = 0.9
    D = data[:int(ratio*len(data))]
    Z_test = data[int(ratio*len(data)):]
    D = [(D[i, :-1], int(D[i, -1])) for i in range(D.shape[0])]
    Z_test = [(Z_test[i, :-1], int(Z_test[i, -1])) for i in range(Z_test.shape[0])]

    sigma = 0.1
    K = 5
    t = 20
    n_clst = len(Z_test) // 10

    topt = shapley_top(D, Z_test, kernel_fn=partial(kernel_value, sigma=sigma), t=t, K=K, n_clst=n_clst)
    # too many after top-9 are equal, so we only check the first 9
    assert topt[0] == 5706
    assert set(topt[1:9]) == set([10495, 17226, 29540, 22601, 35218, 10265, 21305, 8716])


def test_shapley():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    z_test = (np.array([0.0]), 1)

    shapley_values = shapley_bf(D, z_test, K=2, kernel_fn=partial(kernel_value, sigma=1))
    answer = [0.8374, 0.0902, -0.0451]

    assert np.allclose(shapley_values, answer, atol=1e-03)

    # Test multiple test points
    Z_test = [
        (np.array([0.0]), 1),
        (np.array([0.0]), 1),
    ]
    shapley_values = shapley_bf(D, Z_test, K=2, kernel_fn=partial(kernel_value, sigma=1))
    answer = [0.8374, 0.0902, -0.0451]

    assert np.allclose(shapley_values, answer, atol=1e-03)