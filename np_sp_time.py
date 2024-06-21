import numpy as np
import scipy as sp
import pandas as pd
import time

file = 'time_analyse.xlsx'
N = [1500, 2000]
     
nb_it = 100

lst = []
data = []
for n in N:
    np_lstsq = np.zeros(nb_it)
    sp_lstsq = np.zeros(nb_it)
    np_solve = np.zeros(nb_it)
    sp_solve = np.zeros(nb_it)
    np_svd = np.zeros(nb_it)
    sp_svd = np.zeros(nb_it)
    np_qr = np.zeros(nb_it)
    sp_qr = np.zeros(nb_it)
    np_pinv = np.zeros(nb_it)
    sp_pinv = np.zeros(nb_it)
    
    for i in range(nb_it):
        X = np.random.randn(n,n)
        beta = np.array([i%2 for i in range(n)])
        y = X@beta
        
        t1 = time.time()
        np.linalg.lstsq(X,y)
        
        t2 = time.time()
        sp.linalg.lstsq(X,y)
        
        t3 = time.time()
        np.linalg.solve(X, y)
        
        t4 = time.time()
        sp.linalg.solve(X, y)
        
        t5 = time.time()
        np.linalg.svd(X)
        
        t6 = time.time()
        sp.linalg.svd(X)

        t7 = time.time()
        np.linalg.qr(X)
        
        t8 = time.time()
        sp.linalg.qr(X)
        
        t9 = time.time()
        np.linalg.pinv(X)
        
        t10 = time.time()
        sp.linalg.pinv(X)
        
        t11 = time.time()
        
        np_lstsq[i] = t2-t1
        sp_lstsq[i] = t3-t2
        np_solve[i] = t4-t3
        sp_solve[i] = t5-t4
        np_svd[i] = t6-t5
        sp_svd[i] = t7-t6
        np_qr[i] = t8-t7
        sp_qr[i] = t9-t8
        np_pinv[i] = t10-t9
        sp_pinv[i] = t11-t10       
        
    data_np_lstsq = {"n": n, "type": "np_lstsq", "mean": np.mean(np_lstsq), "std": np.std(np_lstsq), "min": np.min(np_lstsq), "max": np.max(np_lstsq)}
    data_sp_lstsq = {"n": n, "type": "sp_lstsq", "mean": np.mean(sp_lstsq), "std": np.std(sp_lstsq), "min": np.min(sp_lstsq), "max": np.max(sp_lstsq)}
    data_np_solve = {"n": n, "type": "np_solve", "mean": np.mean(np_solve), "std": np.std(np_solve), "min": np.min(np_solve), "max": np.max(np_solve)}
    data_sp_solve = {"n": n, "type": "sp_solve", "mean": np.mean(sp_solve), "std": np.std(sp_solve), "min": np.min(sp_solve), "max": np.max(sp_solve)}
    data_np_svd = {"n": n, "type": "np_svd", "mean": np.mean(np_svd), "std": np.std(np_svd), "min": np.min(np_svd), "max": np.max(np_svd)}
    data_sp_svd = {"n": n, "type": "sp_svd", "mean": np.mean(sp_svd), "std": np.std(sp_svd), "min": np.min(sp_svd), "max": np.max(sp_svd)}
    data_np_qr = {"n": n, "type": "np_qr", "mean": np.mean(np_qr), "std": np.std(np_qr), "min": np.min(np_qr), "max": np.max(np_qr)}
    data_sp_qr = {"n": n, "type": "sp_qr", "mean": np.mean(sp_qr), "std": np.std(sp_qr), "min": np.min(sp_qr), "max": np.max(sp_qr)}
    data_np_pinv = {"n": n, "type": "np_pinv", "mean": np.mean(np_pinv), "std": np.std(np_pinv), "min": np.min(np_pinv), "max": np.max(np_pinv)}
    data_sp_pinv = {"n": n, "type": "sp_pinv", "mean": np.mean(sp_pinv), "std": np.std(sp_pinv), "min": np.min(sp_pinv), "max": np.max(sp_pinv)}
    
    lst.append(data_np_lstsq)
    lst.append(data_sp_lstsq)
    lst.append(data_np_solve)
    lst.append(data_sp_solve)
    lst.append(data_np_svd)
    lst.append(data_sp_svd)
    lst.append(data_np_qr)
    lst.append(data_sp_qr)
    lst.append(data_np_pinv)
    lst.append(data_sp_pinv)
    
    
for i in range(10):
    sub_lst = lst[i::10]
    for x in sub_lst:   
        data.append(x)
        
df = pd.DataFrame(data)
df.to_excel(file, index=False)