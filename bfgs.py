import numpy as np


def newton(X, y, p, beta):
    n, m = X.shape
    flag = False
    alpha = 0 
    c1 = 1e-4
    c2 = 0.9
    
    while not(flag):
        err = y - X @ beta
        xp = X@p
        step = -(xp.T@(err - alpha*xp))/(xp.T@xp)
        alpha -= step
        
        if (y- X@(beta-alpha*p)).T@(y- X@(beta-alpha*p)) > err.T@err - 2*c1*alpha*p.T@X.T @ err:
            if 2*p.T@X.T @(err-alpha*X@p) < 2*c2*p.T@X.T @ err:
                flag = True

    return alpha

def bfgs(X, y, num_it):
    n, p = X.shape
    beta = np.zeros(p)
    cost = []

    H_inv = np.eye(p)
    
    for i in range(num_it):
        err = y - X @ beta
        cost.append((1/n) * err.T@err)
        grad = -2/n * X.T @ err

        P = - H_inv @ grad
        
        a = newton(X, y, P, beta)
        
        s = a*P
        
        beta_new = beta + s
        Y = -2/n *X.T@X@(beta_new - beta)
        
        H_inv = H_inv + (s.T@Y+Y.T@H_inv@Y)*(np.reshape(s,(-1, 1))@np.reshape(s,(1, -1)))/(s.T@Y)**2 - (np.reshape(H_inv@Y,(-1, 1))@np.reshape(s,(1, -1))+np.reshape(s,(-1, 1))@np.reshape(Y,(1, -1))@H_inv)/(s.T@Y)

        beta = beta_new
        
    
    return beta, cost

seed = 42
n = 1000
p = 100

X = np.random.normal(0, 1, (n, p))
beta = np.array([i%2 for i in range(p)])
e =  np.random.randn(n)
y = X@beta + e

beta, cost = bfgs(X, y, 15)
print(beta, cost[-1])