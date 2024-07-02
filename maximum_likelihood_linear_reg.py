import numpy as np

def gaussian_noise_likelihood(X, y, alpha = 1e-3, epsilon = 1e-5, max_it=10000):
    n, p = X.shape
    flag = False
    mu = 0
    sigma = 1
    beta = np.zeros(p)
    iteration = 0
    
    while not(flag) and iteration < max_it:
        
        iteration +=1
        
        grad_mu = (np.sum(y-X@beta)-n*mu)/sigma**2
        new_mu = mu + alpha*grad_mu
        
        grad_sigma = np.sum((y-X@beta-mu)**2)/(sigma**3)-n/sigma
        new_sigma = sigma + alpha*grad_sigma
        
        
        grad_beta = X.T@(y-X@beta-mu)/sigma**2
        new_beta = beta + alpha*grad_beta
        
        if np.abs(new_mu-mu)<epsilon and np.abs(new_sigma-sigma)<epsilon and np.sum(np.abs(new_beta-beta))/p<epsilon:
            flag = True
        
        mu = new_mu
        sigma = new_sigma
        beta = new_beta
        
    return mu, sigma, beta

def exp_noise_likelihood(X, y, alpha = 1e-3, epsilon = 1e-5, max_it=10000):
    n, p = X.shape
    flag = False
    lamb = 1
    beta = np.zeros(p)
    
    iteration = 0
    
    while not(flag) and iteration < max_it:
        
        iteration +=1

        grad_lamb = n/lamb - np.sum(y-X@beta)
        new_lamb = lamb + alpha*grad_lamb
        
        grad_beta = lamb * X.T@(y-X@beta)
        new_beta = beta + alpha*grad_beta
        
        if np.abs(new_lamb-lamb)<epsilon and np.sum(np.abs(new_beta-beta))/p<epsilon:
            flag = True
            
        lamb = new_lamb
        beta = new_beta
    
    return lamb, beta

n = 1000
p= 10
X = np.random.normal(0, 1, (n,p))
noise = np.random.exponential(2, n)
beta = np.array([i%2 for i in range(p)])
y = X@beta + noise

mu_est, beta_est = exp_noise_likelihood(X,y,1e-4, 1e-5, 1e5)

print(mu_est, beta_est)