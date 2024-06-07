import numpy as np
import scipy as sp
from gendata import gendata
import time

seed = 42
n = 1000
p = 1000

X, y = gendata(n, p, seed)

t1 = time.time()
Q1, R1 = np.linalg.qr(X)

t2 = time.time()
Q2, R2, = sp.linalg.qr(X)

t3 = time.time()

print(t2-t1, t3-t2)
