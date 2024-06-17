import numpy as np
import scipy as sp
from gendata import gendata
import time

seed = 42
n = 2000
p = 1000

rng = np.random.default_rng(seed)
X = rng.exponential(4, (n, p))
X[:,0] = 1
beta = np.array([i%2 for i in range(p)])
e = rng.exponential(2 , n)
y = X@beta + rng.normal(0, 1, n)

print(np.linalg.cond(X))


#X1 = np.linalg.inv(X)

X2 = np.linalg.pinv(X)

X3 = np.linalg.inv(X.T@X)@X.T

#Q, R = np.linalg.qr(X)
#X4 = sp.linalg.solve_triangular(R, np.eye(n))@Q.T


#L, U = sp.linalg.lu_factor(X)
#X5 = sp.linalg.lu_solve((L, U), np.eye(n))


#print(np.sum((X1@y - beta)**2))
print(np.sum((X2@y - beta)**2))
print(np.sum((X3@y - beta)**2))
#print(np.sum((X4@y - beta)**2))
#print(np.sum((X5@y - beta)**2))