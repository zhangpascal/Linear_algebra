import numpy as np
import scipy as sp
import time

seed = 42
n= 1000
p= 1000

rng = np.random.default_rng(seed)
X = rng.normal(0, 1, (n, p))
beta = np.array([i for i in range(p)])
e = rng.exponential(2 , n)
y = X@beta + e

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normed = (X-X_mean)/X_std

y_centered = y-np.mean(y)

t1 = time.time()
beta_est_ref_np = np.linalg.solve(X_normed, y_centered)

t2 = time.time()
beta_est_ref_sp = sp.linalg.solve(X_normed, y_centered)

t3 = time.time()


print(t2-t1, t3-t2)

print(np.sum(y_centered-X_normed@beta_est_ref_np), np.sum(y_centered-X_normed@beta_est_ref_sp))