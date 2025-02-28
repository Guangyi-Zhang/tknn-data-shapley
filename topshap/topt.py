import numpy as np
from collections import namedtuple, defaultdict
import time
import heapq

from topshap.helper import distance, kernel_value


Point = namedtuple('Point', ['x', 'y', 'idx', 'is_test'])


def kcenter(Z_test, n_clst):
    """
    Cluster the test points into n_clst clusters by k-center algorithm.
    """
    # Use the furthest first (k-center) algorithm
    # Speed up by tracking the minimum distance over already chosen centers for each point, and push them into a max heap
    c_init = 0
    centers_idx = [c_init] # Choose the first test point as the initial center.

    # Initialize the heap
    H = []
    num_points = len(Z_test)
    for i in range(num_points):
        d = distance(Z_test[i][0], Z_test[c_init][0])
        H.append((-d, i, len(centers_idx)))
    heapq.heapify(H)

    # Select additional centers until reaching n_clst (or all points if fewer)
    while len(centers_idx) < min(n_clst, num_points):
        while True:
            d, i, nc = heapq.heappop(H)
            n_diff = len(centers_idx) - nc
            if n_diff > 0:
                d = min(-d, min([distance(Z_test[i][0], Z_test[c][0]) for c in centers_idx[-n_diff:]]))
            else:
                d = -d
            heapq.heappush(H, (-d, i, len(centers_idx)))
            if np.isclose(d, -H[0][0]):
                break
                
        d, i, _ = heapq.heappop(H)
        centers_idx.append(i)
    
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

    def build_landmarks(self, clusters, testidx2center, K):
        """
        Compute distances to each landmark and sort the augmented list.
        """
        self.testidx2center = testidx2center
        for idx_center, cluster in clusters.items():
            landmark = self.Z_test[idx_center][0]
            distances = [distance(x, landmark) for x, _, _, _ in self.augmented]
            sorted_inds = np.argsort(distances)
            sorted_aug = [self.augmented[i] for i in sorted_inds]

            for test_idx, _ in cluster:
                self.testidx2aug[test_idx] = sorted_aug

            # Create data index mapping and test positions
            for idx, z in enumerate(sorted_aug):
                if z.is_test and testidx2center[z.idx][0] == idx_center:
                    self.testidx2augidx[z.idx] = idx

            # Compute Shapley values for landmarks by recursive formula
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
            idx_center, dist_center = self.testidx2center[test_idx]
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


def shapley_top(D, Z_test, t, K, kernel_fn, n_clst=25, i_start=64, tol_topt=1e-3, tol_ball=1e-6):
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
    
    # Cluster the test points into n_clst clusters by k-center algorithm
    start = time.time()
    clusters, testidx2center = kcenter(Z_test, n_clst)
    end = time.time()
    print(f"kcenter took {end - start:.2f} seconds")

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
        top_t_idx = np.argsort(lbs_point)[-t:] 
        top_t_lb = np.min(lbs_point[top_t_idx])
        top_t_idx_set = set(top_t_idx)
        sorted_ub_idx = np.argsort(ups_point) 
        for j in range(min(t+1, len(sorted_ub_idx))): # TODO: corner case t > len(D)
            top_1_ub_idx = sorted_ub_idx[-j-1]
            if top_1_ub_idx in top_t_idx_set:
                continue
            else:
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
