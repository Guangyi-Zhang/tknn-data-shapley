import numpy as np
from collections import namedtuple, defaultdict
import time
import heapq

from topshap.helper import distance, kernel_value
from topshap.naive import shapley_bf_single


Point = namedtuple('Point', ['x', 'y', 'idx', 'is_test'])


def random_center(Z_test, n_clst):
    """
    Randomly select n_clst test points as centers.
    """
    centers_idx = np.random.choice(len(Z_test), n_clst, replace=False)
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


def kcenter_naive(Z_test, n_clst):
    """
    Cluster the test points into n_clst clusters by k-center algorithm.
    """
    # Use the furthest first (k-center) algorithm:
    centers_idx = [0] # Choose the first test point as the initial center.
    # Select additional centers until reaching n_clst (or all points if fewer)
    num_points = len(Z_test)
    dist_min_to_centers = [float('inf')] * num_points
    while len(centers_idx) < min(n_clst, num_points):
        max_dist = -1
        next_center_idx = None
        for i in range(num_points):
            # Compute distance from Z_test[i] to its nearest already-chosen center.
            d = min([distance(Z_test[i][0], Z_test[c][0]) for c in centers_idx])
            if d > max_dist:
                max_dist = d
                next_center_idx = i
        if next_center_idx is not None:
            centers_idx.append(next_center_idx)
    
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
        clusters[best_center].append(i)
        testidx2center[i] = best_center

    return clusters, testidx2center


def kcenter(Z_test, n_clst, be_robust=False, min_radius=None):
    """
    Cluster the test points into n_clst clusters by k-center algorithm.

    When be_robust is True, if a new center induces a tiny cluster, then ignore it and doesn't increment the center count.
    """
    # Use the furthest first (k-center) algorithm
    centers_idx = [0] # Choose the first test point as the initial center.
    testidx2center = {i: 0 for i in range(len(Z_test))} # assign each test point to the nearest center, default to the first center
    # Select additional centers until reaching n_clst (or all points if fewer)
    num_points = len(Z_test)
    dist_min_to_centers = [float('inf')] * num_points
    n_small = 0
    map_small_center_to_prev = dict()
    while len(centers_idx) - n_small < min(n_clst, num_points) + 1:
        max_dist = -1
        next_center_idx = None
        cnt = 0
        prev = testidx2center[centers_idx[-1]]
        for i in range(num_points):
            # Compute distance from Z_test[i] to its nearest already-chosen center.
            d_prev = dist_min_to_centers[i]
            d = distance(Z_test[i][0], Z_test[centers_idx[-1]][0])
            if d < d_prev:
                testidx2center[i] = centers_idx[-1] # re-assign
                dist_min_to_centers[i] = d
                cnt += 1
            d = min(d, d_prev)
            if d > max_dist:
                max_dist = d
                next_center_idx = i
        if be_robust and cnt < max(1, len(Z_test) / n_clst / 10): # less than 10% of the average cluster size
            n_small += 1
            map_small_center_to_prev[centers_idx[-1]] = prev
        if len(centers_idx) - n_small == min(n_clst, num_points): 
            # one more round to get testidx2center right
            break
        if min_radius is not None and max_dist < min_radius:
            break
        if next_center_idx is not None:
            centers_idx.append(next_center_idx)
    
    # Form clusters
    if be_robust:
        # merge small clusters into its previous center
        for center_idx in centers_idx:
            if center_idx not in map_small_center_to_prev:
                continue
            while map_small_center_to_prev[center_idx] in map_small_center_to_prev:
                prev_center_idx = map_small_center_to_prev[center_idx]
                map_small_center_to_prev[center_idx] = map_small_center_to_prev[prev_center_idx]

        clusters = defaultdict(list)
        for test_idx, center_idx in testidx2center.items():
            if center_idx not in map_small_center_to_prev:
                clusters[center_idx].append(test_idx)
            else:
                clusters[map_small_center_to_prev[center_idx]].append(test_idx)
                testidx2center[test_idx] = map_small_center_to_prev[center_idx]
    else:
        clusters = {ci: [] for ci in centers_idx}
        for test_idx, center_idx in testidx2center.items():
            clusters[center_idx].append(test_idx)
    
    return clusters, testidx2center


def kmeans(Z_test, n_clst):
    """
    Cluster the test points into n_clst clusters by sklearn k-means++ algorithm.
    Return the clusters and the testidx2center mapping.
    """
    from sklearn.cluster import KMeans
    
    # Extract feature vectors from test points
    X = np.array([z[0] for z in Z_test])
    
    # Apply KMeans with k-means++ initialization (which is the default)
    kmeans = KMeans(n_clusters=min(n_clst, len(X)), init='k-means++', random_state=0)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # Find the test point index closest to each cluster center
    center_indices = []
    for center in centers:
        distances = [distance(center, Z_test[i][0]) for i in range(len(Z_test))]
        center_idx = np.argmin(distances)
        center_indices.append(center_idx)
    
    # Assign each test point to the nearest center
    clusters = {ci: [] for ci in center_indices}
    testidx2center = {}
    
    for i, label in enumerate(labels):
        cluster_center_idx = center_indices[label]
        clusters[cluster_center_idx].append(i)
        testidx2center[i] = cluster_center_idx
    
    return clusters, testidx2center


def shapley_tknn(D, Z_test, K, radius, kernel_fn, n_clst=10):
    """
    Compute Shapley values for truncated KNN using k-center clustering.
    """
    if not isinstance(Z_test, list):
        Z_test = [Z_test]
    
    # stack D and Z_test
    #Z_test_D = [z for idx, z in enumerate(Z_test)] + [z for idx, z in enumerate(D)]

    # k-center clustering over D and Z_test
    start_time = time.process_time()
    #clusters, testidx2center = kcenter(Z_test_D, n_clst, be_robust=True)
    #clusters, testidx2center = kcenter(Z_test_D, n_clst, min_radius=2*radius)
    #clusters, testidx2center = kmeans(Z_test_D, n_clst)
    clusters, testidx2center = kcenter(Z_test, n_clst)

    # assign each data point to the nearest cluster center
    clusters_Didx = {ci: [] for ci in clusters.keys()}
    centers = [ci for ci in clusters.keys()]
    for i in range(len(D)):
        dists = [distance(D[i][0], Z_test[ci][0]) for ci in clusters.keys()]
        ci = np.argmin(dists)
        clusters_Didx[centers[ci]].append(i)
    clusters_D = {ci: [D[i] for i in cluster] for ci, cluster in clusters_Didx.items()}

    runtime_kcenter = time.process_time() - start_time
    print(f"kcenter/kmeans took {runtime_kcenter:.2f} seconds with #clusters={len(clusters)}")

    # Create a new clusters_D from clusters, by dropping the test points, and recover the true data_idx
    #clusters_Didx = {ci: [(i - len(Z_test)) for i in cluster if i >= len(Z_test)] for ci, cluster in clusters.items()}
    #clusters_D = {ci: [D[i] for i in cluster] for ci, cluster in clusters_Didx.items()}
    print(f"sizes of clusters: {[len(cluster) for cluster in clusters_D.values()]} and ratio={sum([len(cluster) for cluster in clusters_D.values()]) / len(D)}")

    # Compute Shapley values for each cluster
    shapley_values = np.zeros(len(D))
    for i in range(len(Z_test)):
        cluster = clusters_D[testidx2center[i]]
        s = shapley_bf_single(cluster, Z_test[i], K, kernel_fn, radius=radius)
        for j, data_idx in enumerate(clusters_Didx[testidx2center[i]]):
            shapley_values[data_idx] += s[j]
    
    return shapley_values / len(Z_test)


class BallExpander:
    """
    Compute lower and upper bounds for each data point by expanding the ball centered at each test point.
    """
    lb_base_sum_fixed, up_base_sum_fixed = 0, 0
    lbs_diff_point_fixed, ups_diff_point_fixed = None, None
    test_stops = None

    D = None
    Z_test = None
    augmented = None
    testidx2aug = {}
    testidx2augidx = dict()

    def __init__(self, D, Z_test, kernel_fn, tol=1e-6):
        """
        Args:
            D: List of training data tuples (x, y)
            Z_test: List of test points (x_test, y_test)
            tol: Tolerance for stopping the expansion of a ball (test point)
        """
        self.lbs_diff_point_fixed, self.ups_diff_point_fixed = np.zeros(len(D)), np.zeros(len(D))
        self.test_stops = [False] * len(Z_test)

        self.D = D
        self.Z_test = Z_test
        self.kernel_fn = kernel_fn
        self.tol = tol

        # Create augmented list with both training and test points
        self.augmented = [Point(*z, idx, True) for idx, z in enumerate(Z_test)] + [Point(*z, idx, False) for idx, z in enumerate(D)]

    def build_landmarks(self, clusters, testidx2center, K, no_scoring=False):
        """
        Compute distances to each landmark and sort the augmented list.
        """
        self.testidx2center = testidx2center
        for idx_center, cluster in clusters.items():
            landmark = self.Z_test[idx_center][0]
            distances = [distance(x, landmark) for x, _, _, _ in self.augmented]
            sorted_inds = np.argsort(distances)
            sorted_aug = [self.augmented[i] for i in sorted_inds]

            for test_idx in cluster:
                self.testidx2aug[test_idx] = sorted_aug

            # Create data index mapping and test positions
            for idx, z in enumerate(sorted_aug):
                if z.is_test and testidx2center[z.idx] == idx_center:
                    self.testidx2augidx[z.idx] = idx

            # Compute Shapley values for landmarks by recursive formula
            if not no_scoring:
                self.test_stops[idx_center] = True # skip the landmark in future ball expansion
                sorted_w = [self.kernel_fn(distances[i]) for i in sorted_inds]
                cnt = 0
                s_prev, w_prev = None, None
                y_test = self.Z_test[idx_center][1]
                for z, w in zip(sorted_aug[::-1], sorted_w[::-1]):
                    if z.is_test:
                        continue

                    i = len(self.D) - cnt
                    w = w * (1 if z.y == y_test else 0)
                    if cnt == 0:
                        s = w * K / i
                    else:
                        s = s_prev + min(K, i) / i * (w - w_prev)

                    self.lbs_diff_point_fixed[z.idx] += s
                    self.ups_diff_point_fixed[z.idx] += s

                    cnt += 1
                    s_prev, w_prev = s, w

    def build_ball_by_radius(self, pt_test, radius, landmark):
        """
        Build a ball of radius (by distance) around z_test along the sorted augmented list.
        
        Args:
            pt_test: Point namedtuple containing test point information
            radius: Maximum distance from test point to include in ball
            landmark: Coordinates of the landmark (center) of the cluster
            
        Returns:
            List of Point objects within the specified radius
        """
        x_test, y_test = pt_test.x, pt_test.y
        sorted_aug = self.testidx2aug[pt_test.idx]
        pos = self.testidx2augidx[pt_test.idx]
        
        points = []
        
        # Get distance from test point to landmark
        dist_test_to_landmark = distance(x_test, landmark)
        distances = []

        # Expand to the left (points before the test point in the sorted list)
        left = pos - 1
        while left >= 0:
            x, y, dataidx, is_test = sorted_aug[left]
            if is_test:
                left -= 1
                continue

            dist_to_landmark = distance(x, landmark)
            
            # Lower bound on distance to test point using triangle inequality
            lb_dist = abs(dist_to_landmark - dist_test_to_landmark)
            
            if lb_dist > radius:
                # If lower bound exceeds radius, we can stop expanding left
                break
                
            # Check actual distance to see if point is within radius
            actual_dist = distance(x, x_test)
            if actual_dist <= radius:
                points.append(Point(x, y, dataidx, False))
                distances.append(actual_dist)
            
            left -= 1
        
        # Expand to the right (points after the test point in the sorted list)
        right = pos + 1
        while right < len(sorted_aug):
            x, y, dataidx, is_test = sorted_aug[right]
            if is_test:
                right += 1
                continue

            dist_to_landmark = distance(x, landmark)
            
            # Lower bound on distance to test point using triangle inequality
            lb_dist = abs(dist_to_landmark - dist_test_to_landmark)
            
            if lb_dist > radius:
                # If lower bound exceeds radius, we can stop expanding right
                break
                
            # Check actual distance to see if point is within radius
            actual_dist = distance(x, x_test)
            if actual_dist <= radius:
                points.append(Point(x, y, dataidx, False))
                distances.append(actual_dist)
            
            right += 1

        return points, distances

    def build_ball(self, pt_test, i, landmark):
        """
        Build a ball of radius i around z_test along the sorted augmented list.
        """
        x_test, y_test = pt_test.x, pt_test.y
        sorted_aug = self.testidx2aug[pt_test.idx]
        pos = self.testidx2augidx[pt_test.idx]
        
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

    def expand(self, i, K):
        n = len(self.D)
        lb_base_sum, up_base_sum = 0, 0
        lbs_diff_point, ups_diff_point = np.zeros(len(self.D)), np.zeros(len(self.D))
        
        for test_idx, z_test in enumerate(self.Z_test):
            if self.test_stops[test_idx]:
                continue

            x_test, y_test = z_test
            idx_center = self.testidx2center[test_idx]
            landmark = self.Z_test[idx_center][0]
            points, dist_radius = self.build_ball(Point(*z_test, test_idx, True), i, landmark)
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
                        w = self.kernel_fn(d)
                        term2_fixed[rank] = w * K / rank / (rank-1) + term2_fixed[rank-1]
                        rank_fixed_max = max(rank_fixed_max, rank)
                else:
                    weight_max_in_ball = max(weight_max_in_ball, self.kernel_fn(d))
            
            # Compute lower bound on weight for points not in ball
            weight_out_of_ball = self.kernel_fn(dist_radius)
            weight_for_lb = max(weight_max_in_ball, weight_out_of_ball)
            #print(f"bar_ball_size={bar_ball_size}, weight_max_in_ball={weight_max_in_ball}, weight_for_lb={weight_for_lb}")
            
            # Compute base upper bound 
            up_base = min(K, bar_ball_size) * weight_out_of_ball / bar_ball_size if bar_ball_size > 0 else weight_out_of_ball
            if weight_out_of_ball < self.tol: # skip the test point in the future for negligible future weight
                self.test_stops[test_idx] = True
            #print(f"ups_base[{test_idx}]={ups_base[test_idx]}")

            # Compute base lower bound 
            j0 = max(K, bar_ball_size + 1)
            lb_base = - weight_for_lb * (1/(j0 - 1) - 1/n)
            #print(f"lbs_base[{test_idx}]={lbs_base[test_idx]}")

            # Aggregate base bounds
            if self.test_stops[test_idx]:
                self.lb_base_sum_fixed += lb_base
                self.up_base_sum_fixed += up_base
            else:
                lb_base_sum += lb_base
                up_base_sum += up_base
            
            # Compute point-specific bounds
            for rank, (d, x, y, dataidx) in enumerate(sorted_ball, 1): # rank is 1-based
                w = self.kernel_fn(d)
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
                if self.test_stops[test_idx]:
                    self.lbs_diff_point_fixed[dataidx] += lb - lb_base
                    self.ups_diff_point_fixed[dataidx] += ub - up_base
                else:
                    lbs_diff_point[dataidx] += lb - lb_base
                    ups_diff_point[dataidx] += ub - up_base
                #print(f"bounds_point[{dataidx}]={(test_idx, lb, ub)}, term2={term2}")
        
        # Update bounds for each data point
        #print(f"lb_base={lb_base}, up_base={up_base}")
        lbs_point, ups_point = np.zeros(len(self.D)), np.zeros(len(self.D))
        for data_idx, z in enumerate(self.D):
            lbs_point[data_idx] = self.lb_base_sum_fixed + lb_base_sum + lbs_diff_point[data_idx] + self.lbs_diff_point_fixed[data_idx]
            ups_point[data_idx] = self.up_base_sum_fixed + up_base_sum + ups_diff_point[data_idx] + self.ups_diff_point_fixed[data_idx]
            #print(f"lbs_point[{data_idx}]={lbs_point[data_idx]}, ups_point[{data_idx}]={ups_point[data_idx]}")

        return lbs_point, ups_point


def shapley_tknn_expand(D, Z_test, K, radius, kernel_fn, center_type="kcenter", n_clst=25):
    """
    Compute truncated KNN Shapley values using landmark-based ball expansion.
    The difference is that we don't guess the ball radius, but expand the ball until a given radius

    TODO: skip those landmarks
    """
    if not Z_test:
        return np.zeros(len(D))
    
    # Cluster the test points into n_clst clusters by k-center algorithm
    start = time.time()
    if center_type == "kcenter":
        clusters, testidx2center = kcenter(Z_test, n_clst)
    elif center_type == "random":
        clusters, testidx2center = random_center(Z_test, n_clst)
    else:
        raise ValueError(f"Invalid center type: {center_type}")
    end = time.time()
    print(f"{center_type} took {end - start:.2f} seconds")

    # Compute distances to landmark (center) and sort
    start = time.time()
    expander = BallExpander(D, Z_test, kernel_fn)
    expander.build_landmarks(clusters, testidx2center, K, no_scoring=True)
    end = time.time()
    print(f"landmark sorting took {end - start:.2f} seconds")
        
    shapley_values = np.zeros(len(D))
    n_cluster = 0
    for i in range(len(Z_test)):
        z_test = Z_test[i]
        landmark = Z_test[testidx2center[i]][0]
        cluster, distances = expander.build_ball_by_radius(Point(*z_test, i, True), radius, landmark)
        # sort cluster by idx, to match the exact order of D, or it may yield discrepancy against shapley_bf_single(D); though not necessary in practice
        # cluster = sorted(cluster, key=lambda x: x.idx)
        cluster_D = [(pt.x, pt.y) for pt in cluster]
        cluster_Didx = [pt.idx for pt in cluster]
        s = shapley_bf_single(cluster_D, z_test, K, kernel_fn, radius=radius, distances=distances)
        for j, data_idx in enumerate(cluster_Didx):
            shapley_values[data_idx] += s[j]
        
        n_cluster += len(cluster)

    print(f"avg n_cluster/D={n_cluster / len(Z_test) / len(D):.6f}")
    
    return shapley_values / len(Z_test)


def shapley_top(D, Z_test, t, K, kernel_fn, t_ub=None, center_type="kcenter", n_clst=25, i_start=64, tol_topt=1e-3, tol_ball=1e-6):
    """
    Compute top-t Shapley values using landmark-based ball expansion.
    
    Args:
        D: List of training data tuples (x, y)
        Z_test: List of test points (x_test, y_test)
        t: Number of top values to retrieve
        K: Number of neighbors for KNN
        n_clst: Number of clusters for k-center algorithm
        i_start: Starting ball radius
        tol: Tolerance between top-t lower bound and top-1 upper bound
        
    Returns:
        Indices of top-t data points in D by descending Shapley values
    """
    if not Z_test:
        return np.zeros(len(D))
    
    # When t_ub is not None, allowed to return any top-t s.t. t <= t_ub
    t_lb = t
    if t_ub is None:
        t_ub = t
    else:
        t_lb = 1
    
    # Cluster the test points into n_clst clusters by k-center algorithm
    start = time.time()
    if center_type == "kcenter":
        clusters, testidx2center = kcenter(Z_test, n_clst)
    elif center_type == "random":
        clusters, testidx2center = random_center(Z_test, n_clst)
    else:
        raise ValueError(f"Invalid center type: {center_type}")
    end = time.time()
    print(f"{center_type} took {end - start:.2f} seconds")

    # Compute distances to landmark (center) and sort
    start = time.time()
    expander = BallExpander(D, Z_test, kernel_fn, tol_ball)
    expander.build_landmarks(clusters, testidx2center, K)
    end = time.time()
    print(f"landmark sorting took {end - start:.2f} seconds")
        
    n = len(D)
    i = min(i_start, len(expander.augmented) // 2) # ball radius
    while i <= len(expander.augmented):
        # Compute lower and upper bounds for each data point by expanding the ball centered at each test point
        lbs_point, ups_point = expander.expand(i, K)
        
        # Compare top-t lower bounds with top-1 upper bound
        # Try every feasible t
        j_start = 0
        sorted_lb_idx = np.argsort(lbs_point)
        sorted_ub_idx = np.argsort(ups_point) 
        for t in range(t_lb, t_ub+1):
            top_t_idx = sorted_lb_idx[-t:] 
            top_t_lb = np.min(lbs_point[top_t_idx])
            top_t_idx_set = set(top_t_idx)
            for j in range(j_start, min(t+1, len(sorted_ub_idx))): # TODO: corner case t > len(D)
                top_1_ub_idx = sorted_ub_idx[-j-1]
                if top_1_ub_idx in top_t_idx_set:
                    continue
                else: # 1st upper bound not in top-t lower bounds
                    j_start = j # j_start only increases as t increases
                    if top_t_lb >= ups_point[top_1_ub_idx] - tol_topt: # found top-t
                        print(f"found top-t at i={i}: top_t_lb={top_t_lb}, top_1_ub={ups_point[top_1_ub_idx]}")
                        return top_t_idx[::-1] # reverse the order and start from the largest
                    else:
                        print(f"i={i}: top_t_lb={top_t_lb}, top_1_ub={ups_point[top_1_ub_idx]}, #stops={sum(expander.test_stops)}")
                        # find the largest index of a point in sorted_ub_idx whose ups_point[idx] <= top_t_lb
                        # for idx in range(len(sorted_ub_idx)-1, -1, -1):
                        #     if ups_point[sorted_ub_idx[idx]] <= top_t_lb:
                        #         print(f"count idx={idx} out of {len(sorted_ub_idx)}: ub={ups_point[sorted_ub_idx[idx]]}")
                        #         break
                        break

        # Continue and double the ball radius
        if i != len(expander.augmented):
            i *= 2 
            i = min(i, len(expander.augmented))
        else:
            break    
