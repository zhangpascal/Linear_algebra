import numpy as np
import scipy as sp
from gendata import gendata
import time
seed = 42
n= 1000
p= 1000

def SVD_inv(A):
    #print(np.linalg.cond(A))
    
    U, S, V = np.linalg.svd(A)

    S_tronc = np.where(S > 1e-10, S, 0)
    S_pseudo_inv = np.zeros_like(S_tronc)
    S_pseudo_inv[S_tronc>0] = 1/S_tronc[S_tronc>0]
    
    return V.T*S_pseudo_inv@U.T
    

X, y = gendata(n, p, seed)

t1 = time.time()
#beta_est = SVD_inv(X_normed.T@X_normed)@X_normed.T@y_centered

t2 = time.time()
beta_est_ref_np = np.linalg.lstsq(X, y, rcond=1e-10)

t3 = time.time()
beta_est_ref_sp = sp.linalg.lstsq(X, y, cond= 1e-10)

t4 = time.time()

print(t2-t1, t3-t2, t4-t3)

print(np.sum(y-X@beta_est_ref_np[0]), np.sum(y-X@beta_est_ref_sp[0]))

