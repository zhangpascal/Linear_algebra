import numpy as np
import scipy as sp
from gendata import gendata
import time

seed = 42
n = 1000
p = 1000

X, y = gendata(n, p, seed)

t1 = time.time()
U1, S1, V1 = np.linalg.svd(X)

t2 = time.time()
U2, S2, V2, = sp.linalg.svd(X)

t3 = time.time()

print(t2-t1, t3-t2)
