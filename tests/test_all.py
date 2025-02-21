import pytest
import numpy as np

from topshap.shapley import shapley_bf, build_ball, Point, shapley_top


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
    
    testidx2augidx = {0: 1}
    landmark = np.array([0.0])
    
    points, dist_radius = build_ball(pt_test, i=1, sorted_aug=sorted_aug,
                                      testidx2augidx=testidx2augidx,
                                      landmark=landmark)
    
    # Verify new points added
    assert len(points) == 2
    new_ids = [p.idx for p in points]
    assert 1 in new_ids
    assert 2 in new_ids
    
    # Verify radius calculation
    assert np.isclose(dist_radius, 1)
    
    # Increase radius i
    points, dist_radius = build_ball(pt_test, i=2, sorted_aug=sorted_aug,
                                      testidx2augidx=testidx2augidx,
                                      landmark=landmark)
    assert len(points) == 3
    assert 3 in [p.idx for p in points]
    assert np.isclose(dist_radius, 1.5)

    # Increase radius i
    points, dist_radius = build_ball(pt_test, i=3, sorted_aug=sorted_aug,
                                      testidx2augidx=testidx2augidx,
                                      landmark=landmark)
    assert len(points) == 3
    assert np.isclose(dist_radius, float('inf'))

    # Another case with more points before test point
    sorted_aug = sorted_aug[::-1] # reverse the order of sorted_aug
    
    testidx2augidx = {0: 3}
    landmark = np.array([2.5])

    new_points, dist_radius = build_ball(pt_test, i=2, sorted_aug=sorted_aug,
                                      testidx2augidx=testidx2augidx,
                                      landmark=landmark)
    assert len(new_points) == 3
    assert np.isclose(dist_radius, 1.5)


def test_shapley_top():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    Z_test = [(np.array([0.0]), 1)]

    top_idx = shapley_top(D, Z_test, t=1, K=2, sigma=1)
    assert top_idx == [0]

    top_idx = shapley_top(D, Z_test, t=2, K=2, sigma=1)
    assert np.all(top_idx == [0, 1])

    top_idx = shapley_top(D, Z_test, t=3, K=2, sigma=1)
    assert top_idx == None # fail to find [0, 1, 2]


def test_shapley_bf():
    D = [
        (np.array([0.5]), 1),
        (np.array([2.0]), 1),
        (np.array([1.0]), 0)
    ]
    z_test = (np.array([0.0]), 1)

    shapley_values = shapley_bf(D, z_test, K=2, sigma=1)
    answer = [0.8374, 0.0902, -0.0451]

    assert np.allclose(shapley_values, answer, atol=1e-03)

    # Test multiple test points
    Z_test = [
        (np.array([0.0]), 1),
        (np.array([0.0]), 1),
    ]
    shapley_values = shapley_bf(D, Z_test, K=2, sigma=1)
    answer = [0.8374, 0.0902, -0.0451]

    assert np.allclose(shapley_values, answer, atol=1e-03)
