import numpy as np
from collections import namedtuple


def distance(x, y):
    """
    Compute the Euclidean distance between two points.
    """
    return np.linalg.norm(x - y)


def kernel_value(d, sigma):
    """
    Compute the Gaussian kernel value for a given distance.
    """
    return np.exp(-d**2 / (2 * sigma**2))    


Point = namedtuple('Point', ['x', 'y', 'idx', 'is_test'])


class Processed:
    def __init__(self):
        self.ids = set()
        self.pts = []


def build_ball(pt_test, i, sorted_aug, testidx2augidx, processed, landmark):
    """
    Build a ball of radius i around z_test along the sorted augmented list.
    """
    x_test, y_test = pt_test.x, pt_test.y
    pos = testidx2augidx[pt_test.idx]
    
    # Get ball boundaries
    start = max(0, pos - i)
    end = min(len(sorted_aug), pos + i + 1) # +1 for exclusive end
    new_points = []
    
    # Collect new data points in expanded ball
    try_first = False
    dist_left, dist_right = 0, 0
    for idx in range(start, end):
        x, y, dataidx, is_test = sorted_aug[idx]
        if not is_test and dataidx not in processed[pt_test.idx].ids:
            processed[pt_test.idx].ids.add(dataidx)
            processed[pt_test.idx].pts.append(Point(x, y, dataidx, False))
            new_points.append(Point(x, y, dataidx, False))

        # Compute a lower bound on dist_to_test for points outside the ball, left part
        if not is_test and not try_first:
            try_first = True
            dist_to_landmark = distance(x, landmark)
            dist_test_to_landmark = distance(x_test, landmark)
            dist_left = abs(dist_to_landmark - dist_test_to_landmark)

    # Compute a lower bound on dist_to_test for points outside the ball, right part
    for idx in range(end-1, pos, -1):
        x, y, dataidx, is_test = sorted_aug[idx]
        if not is_test:
            dist_to_landmark = distance(x, landmark)
            dist_test_to_landmark = distance(x_test, landmark)
            dist_right = abs(dist_to_landmark - dist_test_to_landmark)
            break
    
    dist_radius = min(dist_left, dist_right)
    return new_points, dist_radius


def shapley_top(D, Z_test, t, K, sigma):
    """
    Compute top-t Shapley values using landmark-based ball expansion.
    
    Args:
        D: List of training data tuples (x, y)
        Z_test: List of test points (x_test, y_test)
        t: Number of top values to retrieve
        K: Number of neighbors for KNN
        sigma: Bandwidth for Gaussian kernel
        
    Returns:
        Indices of top-t data points in D by Shapley value
    """
    if not Z_test:
        return np.zeros(len(D))
    
    # Select first test point as landmark
    landmark = Z_test[0][0]
    
    # Create augmented list with test markers
    augmented = [Point(*z, idx, True) for idx, z in enumerate(Z_test)] + [Point(*z, idx, False) for idx, z in enumerate(D)]
    
    # Compute distances to landmark and sort
    distances = [distance(x, landmark) for x, _, _, _ in augmented]
    sorted_inds = np.argsort(distances)
    sorted_aug = [augmented[i] for i in sorted_inds]
    
    # Create data index mapping and test positions
    testidx2augidx = dict([(z.idx, idx) for idx, z in enumerate(sorted_aug) if z.is_test])
        
    # Initialize bounds
    n = len(D)
    processed = [Processed() for _ in Z_test] # Track processed data per test point
    lbs_base, ups_base = np.zeros(len(Z_test)), np.zeros(len(Z_test)) # base bounds
    bounds_point = [[] for _ in D] # each entry in [] is (test_idx, lower_bound, upper_bound)

    i = 1 # ball radius
    while i <= len(sorted_aug):
        
        for test_idx, z_test in enumerate(Z_test):
            x_test, y_test = z_test
            new_points, dist_radius = build_ball(Point(*z_test, test_idx, True), i, sorted_aug, 
                                                 testidx2augidx, processed, landmark)

            # Compute distances to current test point
            dists = [(distance(x, x_test), x, y, dataidx) for x, y, dataidx, _ in processed[test_idx].pts]
            sorted_ball = sorted(dists, key=lambda x: x[0])

            # Compute |barB_i(z_test)| for points whose local rank equals global rank
            bar_ball_size = 0
            weight_max_in_ball = 0
            for d, x, y, dataidx in sorted_ball:
                if d <= dist_radius:
                    bar_ball_size += 1
                else:
                    weight_max_in_ball = max(weight_max_in_ball, kernel_value(d, sigma))
            
            # Compute lower bound on weight for points not in ball
            weight_out_of_ball = kernel_value(dist_radius, sigma)
            weight_for_lb = max(weight_max_in_ball, weight_out_of_ball)
            
            # Compute base upper bound 
            ups_base[test_idx] = min(K, bar_ball_size) * weight_out_of_ball / bar_ball_size if bar_ball_size > 0 else weight_out_of_ball

            # Compute base lower bound 
            j0 = max(K, bar_ball_size + 1)
            lbs_base[test_idx] = - weight_for_lb * (1/(j0 - 1) - 1/n)
            
            # Compute point-specific bounds
            for rank, (d, x, y, dataidx) in enumerate(sorted_ball, 1): # rank is 1-based
                w = kernel_value(d, sigma)
                j0 = max(K, rank + 1)
                ub = min(K, rank) * w * (1 if y == y_test else 0) / rank if rank > 0 else w * (1 if y == y_test else 0)
                if d <= dist_radius:
                    lb = ub - w * (1/(j0 - 1) - 1/n) # TODO: improve the 2nd term
                else:
                    rank_ = rank + n - bar_ball_size
                    term1 = min(K, rank_) * w * (1 if y == y_test else 0) / rank_ if rank_ > 0 else w * (1 if y == y_test else 0)
                    lb = term1 - weight_for_lb * (1/(j0 - 1) - 1/n) 

                bounds_point[dataidx].append((test_idx, lb, ub))
        
        # Aggregate the base bounds
        lb_base, up_base = 0, 0
        for test_idx, z_test in enumerate(Z_test):
            lb_base += lbs_base[test_idx]
            up_base += ups_base[test_idx]

        # Update bounds for each data point
        lbs_point, ups_point = np.zeros(len(D)), np.zeros(len(D))
        for data_idx, z in enumerate(D):
            lbs_point[data_idx] = lb_base
            ups_point[data_idx] = up_base
            for test_idx, lb, ub in bounds_point[data_idx]:
                lbs_point[data_idx] += lb - lbs_base[test_idx]
                ups_point[data_idx] += ub - ups_base[test_idx]

        # Compare top-t lower bounds with top-1 upper bound
        top_t_idx = np.argsort(lbs_point)[-t:] 
        top_t_lb = np.min(lbs_point[top_t_idx])
        top_t_idx_set = set(top_t_idx)
        sorted_ub_idx = np.argsort(ups_point) 
        for j in range(t+1):
            top_1_ub_idx = sorted_ub_idx[-j-1]
            if top_1_ub_idx in top_t_idx_set:
                continue
            else:
                if top_t_lb >= ups_point[top_1_ub_idx]: # found top-t
                    print(f"found top-t at i={i}: top_t_lb={top_t_lb}, top_1_ub={ups_point[top_1_ub_idx]}")
                    return top_t_idx
                else:
                    break

        # Continue and double the ball radius
        if i != len(sorted_aug):
            i *= 2 
            i = min(i, len(sorted_aug))
        else:
            break


def shapley_bf(D, Z_test, K, sigma):
    """
    Compute Shapley values for weighted KNN using brute force.
    """
    if not isinstance(Z_test, list):
        return shapley_bf_single(D, Z_test, K, sigma)
    
    n_test = len(Z_test)
    shapley_values = np.zeros(len(D))
    for i in range(n_test):
        s = shapley_bf_single(D, Z_test[i], K, sigma)
        shapley_values += s

    return shapley_values / n_test
    

def shapley_bf_single(D, z_test, K, sigma):
    """
    Compute Shapley values for weighted KNN using recursive formula.
    
    Args:
        D: List of tuples (x, y) where x is feature vector, y is label
        z_test: Test point tuple (x_test, y_test)
        K: Number of neighbors for KNN
        sigma: Bandwidth for Gaussian kernel
        
    Returns:
        Array of Shapley values for each data point
    """
    x_test, y_test = z_test
    n = len(D)
    if n == 0:
        return np.array([])
    
    # Calculate distances and sort
    dxy = [(distance(x, x_test), x, y) for x, y in D]
    sorted_dxy_idx = sorted(range(len(dxy)), key=lambda i: dxy[i][0]) # argsort
    
    # Extract weights and label matches
    w = [kernel_value(d, sigma) for d, _, _ in dxy]
    y_match = [1 if y == y_test else 0 for _, _, y in dxy]
    
    # Initialize Shapley values array
    s = np.zeros(n)
    
    # Base case: farthest point
    idx_n = sorted_dxy_idx[-1]
    s[idx_n] = (K/n) * w[idx_n] * y_match[idx_n]
    
    # Recursive calculation from 2nd farthest to nearest
    for j in range(n-2, -1, -1):
        i_plus_1 = j + 1  # Convert to 1-based index
        idx_j = sorted_dxy_idx[j]
        idx_j_plus_1 = sorted_dxy_idx[j+1]
        term = (min(K, i_plus_1)/i_plus_1) * (
            w[idx_j] * y_match[idx_j] - w[idx_j_plus_1] * y_match[idx_j_plus_1]
        )
        s[idx_j] = s[idx_j_plus_1] + term
        
    return s 