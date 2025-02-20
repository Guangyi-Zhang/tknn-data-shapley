import pytest
import numpy as np

from topshap.shapley import shapley_bf, build_ball, Point, Processed


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
    processed = {0: Processed()}
    landmark = np.array([0.0])
    
    new_points, dist_radius = build_ball(pt_test, i=1, sorted_aug=sorted_aug,
                                      testidx2augidx=testidx2augidx,
                                      processed=processed, landmark=landmark)
    
    # Verify new points added
    assert len(new_points) == 2
    new_ids = [p.idx for p in new_points]
    assert 1 in new_ids
    assert 2 in new_ids
    
    # Verify radius calculation
    assert np.isclose(dist_radius, 0.5)
    
    # Verify processed state updated
    assert len(processed[0].pts) == 2
    
    # Increase radius i
    new_points, dist_radius = build_ball(pt_test, i=2, sorted_aug=sorted_aug,
                                      testidx2augidx=testidx2augidx,
                                      processed=processed, landmark=landmark)
    assert len(new_points) == 1
    assert 3 in [p.idx for p in new_points]
    assert np.isclose(dist_radius, 0.5)

    # Increase radius i
    new_points, dist_radius = build_ball(pt_test, i=3, sorted_aug=sorted_aug,
                                      testidx2augidx=testidx2augidx,
                                      processed=processed, landmark=landmark)
    assert len(new_points) == 0
    assert np.isclose(dist_radius, 0.5)

    #TODO: another case with more points before test point


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
