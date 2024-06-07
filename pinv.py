import numpy as np
import scipy as sp
from gendata import gendata
import time

seed = 42
n = 1000
p = 500

def SVD_inv(A):
    #print(np.linalg.cond(A))
    
    U, S, V = np.linalg.svd(A)

    S_tronc = np.where(S > 1e-10, S, 0)
    S_pseudo_inv = np.zeros_like(S_tronc)
    S_pseudo_inv[S_tronc>0] = 1/S_tronc[S_tronc>0]

    
    return V.T*S_pseudo_inv@U.T
    
X, y = gendata(n, p, seed)

t1 = time.time()
#A1 = SVD_inv(X)

t2 = time.time()
A2 = np.linalg.pinv(X)

t3 = time.time()
A3 = sp.linalg.pinv(X)

t4 = time.time()

print(t2-t1, t3-t2, t4-t3)
