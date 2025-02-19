import numpy as np


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
    augmented = [(*z, idx, True) for idx, z in enumerate(Z_test)] + [(*z, idx, False) for idx, z in enumerate(D)]
    
    # Compute distances to landmark and sort
    distances = [np.linalg.norm(x - landmark) for x, _, _, _ in augmented]
    sorted_inds = np.argsort(distances)
    sorted_aug = [augmented[i] for i in sorted_inds]
    
    # Create data index mapping and test positions
    testidx2augidx = dict()
    for idx, z in enumerate(sorted_aug):
        if z[3]:
            testidx2augidx[z[2]] = idx
        
    # Initialize bounds
    n = len(D)
    lower = np.zeros(n)
    upper = np.zeros(n)
    processed_ids = [set() for _ in Z_test]  # Track processed data per test point
    processed = [[] for _ in Z_test]
    lbs_base, ups_base = np.zeros(len(Z_test)), np.zeros(len(Z_test))
    bounds_point = [[] for _ in D] # each entry in [] is (test_idx, lower_bound, upper_bound)
    
    i = 1 # ball radius
    while i <= len(sorted_aug):
        
        for test_idx, z_test in enumerate(Z_test):
            x_test, y_test = z_test
            pos = testidx2augidx[test_idx]
            
            # Get ball boundaries
            start = max(0, pos - i)
            end = min(len(sorted_aug), pos + i + 1) # +1 for exclusive end
            new_points = []
            
            # Collect new data points in expanded ball
            try_first, try_last = False, False
            dist_left, dist_right = 0, 0
            for idx in range(start, end):
                x, y, dataidx, is_test = sorted_aug[idx]
                if not is_test and dataidx not in processed_ids[test_idx]:
                    processed_ids[test_idx].add(dataidx)
                    processed[test_idx].append((x, y, dataidx))
                    new_points.append((x, y, dataidx))

                # Compute a lower bound on dist_to_test for points outside the ball
                if not is_test and not try_first:
                    try_first = True
                    dist_to_landmark = np.linalg.norm(x - landmark)
                    dist_test_to_landmark = np.linalg.norm(x_test - landmark)
                    dist_left = abs(dist_to_landmark - dist_test_to_landmark)
                if not is_test and not try_last:
                    try_last = True
                    dist_to_landmark = np.linalg.norm(x - landmark)
                    dist_test_to_landmark = np.linalg.norm(x_test - landmark)
                    dist_right = abs(dist_to_landmark - dist_test_to_landmark)

            # Skip if no new points
            if not new_points:
                continue

            # Compute distances to current test point
            dists = [(np.linalg.norm(x - x_test), x, y, dataidx) for x, y, dataidx in processed[test_idx]]
            sorted_ball = sorted(dists, key=lambda x: x[0])

            # Compute |barB_i(z_test)| for points whose local rank equals global rank
            bar_ball_size = 0
            dist_radius = min(dist_left, dist_right)
            kernel_val_max_in_ball = 0
            for d, x, y, dataidx in sorted_ball:
                if d <= dist_radius:
                    bar_ball_size += 1
                else:
                    kernel_val_max_in_ball = max(kernel_val_max_in_ball, np.exp(-d**2 / (2 * sigma**2)))
            
            # Compute lower bound on weight for points not in ball
            kernel_val_out_of_ball = np.exp(-dist_radius**2 / (2 * sigma**2))
            kernel_val_for_bound = max(kernel_val_max_in_ball, kernel_val_out_of_ball)
            
            # Compute base upper bound 
            ups_base[test_idx] = min(K, bar_ball_size) * kernel_val_out_of_ball / bar_ball_size if bar_ball_size > 0 else kernel_val_out_of_ball

            # Compute base lower bound 
            j0 = max(K, bar_ball_size + 1)
            lbs_base[test_idx] = - kernel_val_for_bound * (1/(j0 - 1) - 1/n)
            
            # Compute point-specific bounds
            for rank, (d, x, y, dataidx) in enumerate(sorted_ball, 1): # rank is 1-based
                w = np.exp(-d**2 / (2 * sigma**2))
                j0 = max(K, rank + 1)
                ub = min(K, rank) * w * (1 if y == y_test else 0) / rank if rank > 0 else w * (1 if y == y_test else 0)
                if d <= dist_radius:
                    lb = ub - w * (1/(j0 - 1) - 1/n) # TODO: improve the 2nd term
                else:
                    rank_ = rank + n - bar_ball_size
                    term1 = min(K, rank_) * w * (1 if y == y_test else 0) / rank_ if rank_ > 0 else w * (1 if y == y_test else 0)
                    lb = term1 - kernel_val_for_bound * (1/(j0 - 1) - 1/n) 

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
        top_1_ub = ups_point[np.argmax(ups_point)]
        if top_t_lb >= top_1_ub: # found top-t
            return top_t_idx
    
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
        Array of Shapley values for each data point in sorted order
    """
    x_test, y_test = z_test
    n = len(D)
    if n == 0:
        return np.array([])
    
    # Calculate distances and sort
    sorted_D = sorted(
        [(np.linalg.norm(x - x_test), x, y) for x, y in D],
        key=lambda x: x[0]
    )
    
    # Extract weights and label matches
    w = [np.exp(-d**2/(2*sigma**2)) for d, _, _ in sorted_D]
    y_match = [1 if y == y_test else 0 for _, _, y in sorted_D]
    
    # Initialize Shapley values array
    s = np.zeros(n)
    
    # Base case: farthest point
    s[-1] = (K/n) * w[-1] * y_match[-1]
    
    # Recursive calculation from 2nd farthest to nearest
    for j in range(n-2, -1, -1):
        i_plus_1 = j + 1  # Convert to 1-based index
        term = (min(K, i_plus_1)/i_plus_1) * (
            w[j] * y_match[j] - w[j+1] * y_match[j+1]
        )
        s[j] = s[j+1] + term
        
    return s 