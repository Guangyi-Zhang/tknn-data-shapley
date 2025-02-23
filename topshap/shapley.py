import numpy as np
from collections import namedtuple, defaultdict


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


def kcenter(Z_test, n_clst):
    """
    Cluster the test points into n_clst clusters by k-center algorithm.
    """
    # Use the furthest first (k-center) algorithm:
    centers_idx = set([0]) # Choose the first test point as the initial center.
    # Select additional centers until reaching n_clst (or all points if fewer)
    num_points = len(Z_test)
    while len(centers_idx) < min(n_clst, num_points):
        max_dist = -1
        next_center_idx = None
        for i in range(num_points):
            if i in centers_idx:
                continue
            # Compute distance from Z_test[i] to its nearest already-chosen center.
            d = min([distance(Z_test[i][0], Z_test[c][0]) for c in centers_idx])
            if d > max_dist:
                max_dist = d
                next_center_idx = i
        if next_center_idx is not None:
            centers_idx.add(next_center_idx)
    
    # Assign each test point to the nearest center, forming clusters.
    clusters = {ci: [] for ci in centers_idx}
    testidx2center = {}
    for i, point in enumerate(Z_test):
        best_center = None
        best_d = float('inf')
        for ci in centers_idx:
            d = distance(point[0], Z_test[ci][0])
            if d < best_d:
                best_d = d
                best_center = ci
        clusters[best_center].append((i, best_d))
        testidx2center[i] = (best_center, best_d)

    return clusters, testidx2center


def build_ball(pt_test, i, sorted_aug, testidx2augidx, landmark):
    """
    Build a ball of radius i around z_test along the sorted augmented list.
    """
    x_test, y_test = pt_test.x, pt_test.y
    pos = testidx2augidx[pt_test.idx]
    
    # Get ball boundaries
    start = max(0, pos - i)
    end = min(len(sorted_aug), pos + i + 1) # +1 for exclusive end
    points = []
    
    # Collect new data points in expanded ball
    for idx in range(start, end):
        x, y, dataidx, is_test = sorted_aug[idx]
        if not is_test:
            points.append(Point(x, y, dataidx, False))

    # Compute a lower bound on dist_to_test for points outside the ball, left part
    dist_left, dist_right = 0, 0
    x, y, dataidx, is_test = sorted_aug[start]
    dist_to_landmark = distance(x, landmark)
    dist_test_to_landmark = distance(x_test, landmark)
    dist_left = abs(dist_to_landmark - dist_test_to_landmark)

    # Compute a lower bound on dist_to_test for points outside the ball, right part
    x, y, dataidx, is_test = sorted_aug[end-1]
    dist_to_landmark = distance(x, landmark)
    dist_test_to_landmark = distance(x_test, landmark)
    dist_right = abs(dist_to_landmark - dist_test_to_landmark)
    
    dist_radius = min(dist_left if start > 0 else float('inf'), 
                      dist_right if end < len(sorted_aug) else float('inf'))
    return points, dist_radius


lb_base_sum_fixed, up_base_sum_fixed = 0, 0
lbs_diff_point_fixed, ups_diff_point_fixed = None, None
test_stops = None

def shapley_top_i(D, Z_test, testidx2center, testidx2aug, testidx2augidx, K, sigma, i, test_tol=1e-6):
    """
    Run shapley_top for a specified landmark and ball radius i.
    Return lb_base_sum, up_base_sum, lbs_diff_point, ups_diff_point
    """
    global lb_base_sum_fixed, up_base_sum_fixed, lbs_diff_point_fixed, ups_diff_point_fixed, test_stops
    if lbs_diff_point_fixed is None:
        lbs_diff_point_fixed = np.zeros(len(D))
        ups_diff_point_fixed = np.zeros(len(D))
        test_stops = [False] * len(Z_test)
    n = len(D)
    lb_base_sum, up_base_sum = 0, 0
    lbs_diff_point, ups_diff_point = np.zeros(len(D)), np.zeros(len(D))
    
    for test_idx, z_test in enumerate(Z_test):
        if test_stops[test_idx]:
            continue
        x_test, y_test = z_test
        idx_center, dist_center = testidx2center[test_idx]
        landmark = Z_test[idx_center][0]
        sorted_aug = testidx2aug[test_idx]
        points, dist_radius = build_ball(Point(*z_test, test_idx, True), i, sorted_aug, 
                                                testidx2augidx, landmark)
        #print(f"i={i}, dist_radius={dist_radius}, processed[{test_idx}].ids={processed[test_idx].ids}")

        # Compute distances to current test point
        dists = [(distance(x, x_test), x, y, dataidx) for x, y, dataidx, _ in points]
        sorted_ball = sorted(dists, key=lambda x: x[0])
        if dist_radius == float('inf') and len(sorted_ball) > 0: # no point is outside the ball
            dist_radius = sorted_ball[-1][0] # set to max dist

        # Compute |barB_i(z_test)| for points whose local rank equals global rank
        bar_ball_size = 0
        weight_max_in_ball = 0
        term2_fixed = defaultdict(int) # keep the accumulated weighted terms
        rank_fixed_max = 0
        for rank, (d, x, y, dataidx) in enumerate(sorted_ball, 1): # rank is 1-based
            if d <= dist_radius:
                bar_ball_size += 1
                if rank > 1:
                    w = kernel_value(d, sigma)
                    term2_fixed[rank] = w * K / rank / (rank-1) + term2_fixed[rank-1]
                    rank_fixed_max = max(rank_fixed_max, rank)
            else:
                weight_max_in_ball = max(weight_max_in_ball, kernel_value(d, sigma))
        
        # Compute lower bound on weight for points not in ball
        weight_out_of_ball = kernel_value(dist_radius, sigma)
        weight_for_lb = max(weight_max_in_ball, weight_out_of_ball)
        #print(f"bar_ball_size={bar_ball_size}, weight_max_in_ball={weight_max_in_ball}, weight_for_lb={weight_for_lb}")
        
        # Compute base upper bound 
        up_base = min(K, bar_ball_size) * weight_out_of_ball / bar_ball_size if bar_ball_size > 0 else weight_out_of_ball
        if weight_out_of_ball < test_tol:
            test_stops[test_idx] = True
        #print(f"ups_base[{test_idx}]={ups_base[test_idx]}")

        # Compute base lower bound 
        j0 = max(K, bar_ball_size + 1)
        lb_base = - weight_for_lb * (1/(j0 - 1) - 1/n)
        #print(f"lbs_base[{test_idx}]={lbs_base[test_idx]}")

        # Aggregate base bounds
        if test_stops[test_idx]:
            lb_base_sum_fixed += lb_base
            up_base_sum_fixed += up_base
        else:
            lb_base_sum += lb_base
            up_base_sum += up_base
        
        # Compute point-specific bounds
        for rank, (d, x, y, dataidx) in enumerate(sorted_ball, 1): # rank is 1-based
            w = kernel_value(d, sigma)
            j0 = max(K, rank + 1)

            term1 = min(K, rank) * w * (1 if y == y_test else 0) / rank
            term2 = term2_fixed[rank_fixed_max] - term2_fixed[j0-1] if rank_fixed_max >= j0 else 0
            ub = term1 - term2

            term2_lb = term2 + weight_for_lb * (1/(max(j0, rank_fixed_max+1) - 1) - 1/n)
            if d <= dist_radius:
                lb = term1 - term2_lb
            else:
                rank_ = rank + n - len(sorted_ball)
                term1 = min(K, rank_) * w * (1 if y == y_test else 0) / rank_ 
                lb = term1 - term2_lb

            # Aggregate differences in the point-specific bounds
            if test_stops[test_idx]:
                lbs_diff_point_fixed[dataidx] += lb - lb_base
                ups_diff_point_fixed[dataidx] += ub - up_base
            else:
                lbs_diff_point[dataidx] += lb - lb_base
                ups_diff_point[dataidx] += ub - up_base
            #print(f"bounds_point[{dataidx}]={(test_idx, lb, ub)}, term2={term2}")
    
    # Update bounds for each data point
    #print(f"lb_base={lb_base}, up_base={up_base}")
    lbs_point, ups_point = np.zeros(len(D)), np.zeros(len(D))
    for data_idx, z in enumerate(D):
        lbs_point[data_idx] = lb_base_sum_fixed + lb_base_sum + lbs_diff_point[data_idx] + lbs_diff_point_fixed[data_idx]
        ups_point[data_idx] = up_base_sum_fixed + up_base_sum + ups_diff_point[data_idx] + ups_diff_point_fixed[data_idx]
        #print(f"lbs_point[{data_idx}]={lbs_point[data_idx]}, ups_point[{data_idx}]={ups_point[data_idx]}")

    return lbs_point, ups_point


def shapley_top(D, Z_test, t, K, sigma, n_clst=25, i_start=1, tol=1e-3):
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
    
    # Cluster the test points into n_clst clusters by k-center algorithm
    clusters, testidx2center = kcenter(Z_test, n_clst)

    # Create augmented list with test markers
    augmented = [Point(*z, idx, True) for idx, z in enumerate(Z_test)] + [Point(*z, idx, False) for idx, z in enumerate(D)]

    # Compute distances to landmark and sort
    testidx2aug = {}
    testidx2augidx = dict()
    for idx_center, cluster in clusters.items():
        landmark = Z_test[idx_center][0]
        distances = [distance(x, landmark) for x, _, _, _ in augmented]
        sorted_inds = np.argsort(distances)
        sorted_aug = [augmented[i] for i in sorted_inds]

        for test_idx, _ in cluster:
            testidx2aug[test_idx] = sorted_aug

        # Create data index mapping and test positions
        for idx, z in enumerate(sorted_aug):
            if z.is_test and testidx2center[z.idx][0] == idx_center:
                testidx2augidx[z.idx] = idx
        
    n = len(D)
    i = i_start # ball radius
    while i <= len(sorted_aug):
        lbs_point, ups_point = shapley_top_i(D, Z_test, testidx2center, testidx2aug, testidx2augidx, K, sigma, i)

        # Compare top-t lower bounds with top-1 upper bound
        top_t_idx = np.argsort(lbs_point)[-t:] 
        top_t_lb = np.min(lbs_point[top_t_idx])
        top_t_idx_set = set(top_t_idx)
        sorted_ub_idx = np.argsort(ups_point) 
        for j in range(min(t+1, len(sorted_ub_idx))): # TODO: corner case t > len(D)
            top_1_ub_idx = sorted_ub_idx[-j-1]
            if top_1_ub_idx in top_t_idx_set:
                continue
            else:
                if top_t_lb >= ups_point[top_1_ub_idx] - tol: # found top-t
                    print(f"found top-t at i={i}: top_t_lb={top_t_lb}, top_1_ub={ups_point[top_1_ub_idx]}")
                    return top_t_idx[::-1] # reverse the order and start from the largest
                else:
                    print(f"i={i}: top_t_lb={top_t_lb}, top_1_ub={ups_point[top_1_ub_idx]}, #stops={sum(test_stops)}")
                    # find the largest index of a point in sorted_ub_idx whose ups_point[idx] <= top_t_lb
                    # for idx in range(len(sorted_ub_idx)-1, -1, -1):
                    #     if ups_point[sorted_ub_idx[idx]] <= top_t_lb:
                    #         print(f"count idx={idx} out of {len(sorted_ub_idx)}: ub={ups_point[sorted_ub_idx[idx]]}")
                    #         break
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