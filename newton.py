import numpy as np

def newton(X, y, num_it):
    n, p = X.shape
    beta = np.zeros(p)
    cost = []
    
    for i in range(num_it):
        err = y - X@beta
        cost.append(1/n * err.T@err)
        grad = -X.T@err
        hess =  X.T@X
        beta -= np.linalg.pinv(hess)@grad
        
    return beta, cost
        

seed = 42
n = 1000
p = 100

#rng = np.random.default_rng(seed)
X = np.random.normal(0, 1, (n, p))
beta = np.array([i%2 for i in range(p)])
e =  np.random.randn(n)
y = X@beta + e

beta, cost = newton(X, y, 1)
print(beta, cost[-1])