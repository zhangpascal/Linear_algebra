import numpy as np

def gendata(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    beta = np.array([i for i in range(p)])
    e = rng.exponential(2 , n)
    y = X@beta + e

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normed = (X-X_mean)/X_std

    y_centered = y-np.mean(y)
    
    return X_normed, y_centered
