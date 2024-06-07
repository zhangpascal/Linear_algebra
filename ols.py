import numpy as np
import scipy as sp
from gendata import gendata
import time

seed = 42
n = 1000
p = 1000

X, y = gendata(n, p, seed)

t1 = time.time()
beta_est_ref_np = np.linalg.lstsq(X, y, rcond=1e-10)

t2 = time.time()
beta_est_ref_sp = sp.linalg.lstsq(X, y, cond= 1e-10)

t3 = time.time()

print(t2-t1, t3-t2)

print(np.sum(y-X@beta_est_ref_np[0]), np.sum(y-X@beta_est_ref_sp[0]))

