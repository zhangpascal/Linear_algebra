import numpy as np

seed = 42
n= 100
p= 50

def SVD_inv(A):
    print(np.linalg.cond(A))
    
    U, S, V = np.linalg.svd(A)

    S_tronc = np.where(S > 1e-10, S, 0)
    S_pseudo_inv = np.zeros_like(S_tronc)
    S_pseudo_inv[S_tronc>0] = 1/S_tronc[S_tronc>0]
    
    return V.T*S_pseudo_inv@U.T
    

rng = np.random.default_rng(seed)
X = rng.exponential( 2, (n, p))
beta = np.array([i for i in range(p)])
e = rng.laplace(200, 1 , n)
y = X@beta + e

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X-X_mean)/X_std

y = y-np.mean(y)

beta_est = SVD_inv(X.T@X)@X.T@y

beta_est_ref = np.linalg.lstsq(X, y)

print(beta_est/X_std, beta_est_ref[0])

print(np.sum(y-X@beta_est), np.sum(y-X@beta_est_ref[0]))