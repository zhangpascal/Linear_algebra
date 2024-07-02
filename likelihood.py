import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def log_likelihood(params, data):
    mu, sigma = params
    return -np.sum(np.log(norm.pdf(data, loc=mu, scale=sigma)))


def gaussian_likelihood(x, alpha = 1e-3, epsilon = 1e-3):
    n = x.size
    flag = False
    sigma = 1
    mu = 0
    
    while not(flag):
        grad_mu = (np.sum(x)-n*mu)/sigma**2
        new_mu = mu + alpha*grad_mu
        
        grad_sigma = np.sum((x-mu)**2)/(sigma**3)-n/sigma
        new_sigma = sigma + alpha*grad_sigma

        if (np.abs(new_mu-mu)<epsilon) and (np.abs(new_sigma-sigma))<epsilon:
            flag = True
        
        mu = new_mu
        sigma = new_sigma
        
    return mu, sigma

def exp_likelihood(x, alpha = 1e-3, epsilon = 1e-3 ):
    n = x.size
    flag = False
    lamb= 1
    
    while not(flag):
        grad_lamb = n/lamb - np.sum(x)
        new_lamb = lamb + alpha*grad_lamb
        
        if (np.abs(new_lamb-lamb)<epsilon):
            flag = True
            
        lamb = new_lamb
    
    return lamb

def log_norm_likelihood(x, alpha = 1e-3, epsilon = 1e-3):
    n = x.size
    flag = False
    sigma = 1
    mu = 0
    
    while not(flag):
        grad_mu = (np.sum(np.log(x))-n*mu)/sigma**2
        new_mu = mu + alpha*grad_mu
        
        grad_sigma = np.sum((np.log(x)-mu)**2)/(sigma**3)-n/sigma
        new_sigma = sigma + alpha*grad_sigma

        if (np.abs(new_mu-mu)<epsilon) and (np.abs(new_sigma-sigma))<epsilon:
            flag = True
        
        mu = new_mu
        sigma = new_sigma
        
    return mu, sigma

def laplacian_likelihood(x, alpha = 1e-3, epsilon = 1e-3):
    n = x.size
    flag = False
    b = 1
    mu = 0
    
    while not(flag):
        grad_mu = np.sum(np.sign(x-mu)) / b
        new_mu = mu + alpha*grad_mu
        
        grad_b = np.sum(np.abs(x-mu))/(b**2)-n/b
        new_b = b + alpha*grad_b

        if (np.abs(new_mu-mu)<epsilon) and (np.abs(new_b-b))<epsilon:
            flag = True
        
        mu = new_mu
        b = new_b

        
    return mu, b

n = 99
x = np.random.laplace(2,4,n)

mu_est, b_est = laplacian_likelihood(x,1e-3,1e-3)


result = minimize(log_likelihood, [0, 1], args=(x,), method='L-BFGS-B', bounds=[(None, None), (1e-6, None)])

mu_mle, sigma_mle = result.x

print(mu_est,b_est)
print(np.median(x), np.sum(np.abs(x-np.median(x)))/n)
            
     
    
    
    
