import numpy as np
import scipy as sp
from gendata import gendata
import time

seed = 42
n = 1000
p = 1000

X = np.random.normal(0,1,(n,n))
X=(X-np.mean(X,axis=0))
#/np.std(X,axis=0)
print(np.linalg.cond(X))

t1 = time.time()
X1 = np.linalg.inv(X)

t2 = time.time()
X2 = np.linalg.pinv(X)

t3 = time.time()
Q, R = np.linalg.qr(X)
X3 = sp.linalg.solve_triangular(R, np.eye(n))@Q.T

t4 = time.time()
L, U = sp.linalg.lu_factor(X)
X4 = sp.linalg.lu_solve((L, U), np.eye(n))

t5 = time.time()


print(t2-t1, t3-t2, t4-t3, t5-t4)

print(np.sum((X1@X - np.eye(n))**2))
print(np.sum((X2@X - np.eye(n))**2))
print(np.sum((X3@X - np.eye(n))**2))
print(np.sum((X4@X - np.eye(n))**2))



