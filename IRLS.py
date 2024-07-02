import numpy as np

def iterative_reweighting_mu(x, tol=1e-6, max_iter=1000):

    mu = np.mean(x)
    sigma = np.std(x)
    epsilon = 1e-8 
    flag = False
    i=0

    while not flag and i < max_iter:

        res = (x - mu)/sigma

        weights = 1 / (np.abs(res) + epsilon)

        mu_new = np.sum(weights * x )/ np.sum(weights)

        if np.abs(mu_new - mu) < tol:
            flag = True
            
        mu = mu_new
        
        i += 1
        
    print(i)

    return mu

def iterative_reweighting_sigma(x, mu,  tol=1e-6, max_iter=10000):
    
    sigma = np.std(x)
    epsilon = 1e-8 
    flag = False
    i=0

    while not flag and i < max_iter:

        res = (x - mu)/sigma

        weights = 1 / (np.abs(res) + epsilon)

        sigma_new = np.sqrt(np.sum(weights * res**2) / np.sum(weights))

        if np.abs(sigma_new- sigma) < tol:
            flag = True
            
        sigma = sigma_new
        
        i += 1
    print(i)

    return sigma




n = 1000
x = np.random.normal(2,4,n)

mu_est = iterative_reweighting_mu(x)
sigma_est = iterative_reweighting_sigma(x, mu_est)

print(mu_est, sigma_est)
