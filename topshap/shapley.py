import numpy as np

def shapley_bf(D, z_test, K, sigma):
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