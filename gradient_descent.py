import numpy as np
from gendata import gendata

def grad_descent(X, y, alpha, num_it):
    n, p = X.shape
    beta = np.zeros(p)
    cost = []
    
    for i in range(num_it):
        err = y - X@beta
        cost.append(1/n * err.T@err)
        grad = -2/n * X.T@err
        beta -= alpha * grad
        
    return beta, cost

seed = 42
n = 100
p = 100

#rng = np.random.default_rng(seed)
X = np.random.normal(0, 1, (n, p))
beta = np.array([i%2 for i in range(p)])
y = X@beta

beta, cost = grad_descent(X, y, 1e-2, 10000)
print(beta, cost[-1])