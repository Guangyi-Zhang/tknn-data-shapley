import numpy as np
import itertools

from topshap.helper import distance, kernel_value


class LSH:
    def __init__(self, n_tables, n_bits):
        self.n_tables = n_tables
        self.n_bits = n_bits

        self.hash_tables = []
        self.random_vecs = []

    def build_hash_tables(self, X):
        for _ in range(self.n_tables):
            # Generate random projection vectors
            vecs = np.random.randn(self.n_bits, X.shape[1])
            self.random_vecs.append(vecs)
            
            # Create hash table for this projection
            table = {}
            for idx, x in enumerate(X):
                hash_key = tuple((x @ vecs.T > 0).astype(int))  # Compute hash bits
                if hash_key not in table:
                    table[hash_key] = []
                table[hash_key].append(idx)
            self.hash_tables.append(table)

    def query(self, x, n_nb):
        """
        Query hash tables to find candidate neighbors.
        Probe nearby hash keys to find more up to n_nb candidates.
        """
        candidates = set()
        
        # Query hash tables to find candidate neighbors
        for table, vecs in zip(self.hash_tables, self.random_vecs):
            hash_key = tuple((x @ vecs.T > 0).astype(int))
            if hash_key in table:
                candidates.update(table[hash_key])

        if len(candidates) >= n_nb:
            return candidates

        # Probe nearby hash keys to find more up to n_nb candidates
        for nflipped in range(1, self.n_bits+1): # try all combinations of flipping bits
            for table, vecs in zip(self.hash_tables, self.random_vecs):
                hash_key = tuple((x @ vecs.T > 0).astype(int))
                for comb in itertools.combinations(range(self.n_bits), nflipped):
                    hash_key_comb = list(hash_key)
                    for i in comb:
                        hash_key_comb[i] = 1 - hash_key_comb[i]
                    hash_key_comb = tuple(hash_key_comb)
                    if hash_key_comb in table:
                        candidates.update(table[hash_key_comb])

                if len(candidates) >= n_nb:
                    break

            if len(candidates) >= n_nb:
                break

        return candidates


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