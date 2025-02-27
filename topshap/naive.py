import numpy as np

from topshap.helper import distance, kernel_value


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