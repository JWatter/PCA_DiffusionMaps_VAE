import numpy as np
from scipy.linalg import sqrtm

def diffusionMaps(X, L): 
    N = X.shape[0]

    # form distance matrix D -> (N, N)
    D = np.empty((N, N))
    for i in range(N):
        for j in range(i, N):
            D[i][j] = np.linalg.norm(X[i] - X[j])
            D[j][j] = D[i][j]
    
    # set epsilon
    epsilon = 0.05 * D.max()
    
    # initialize kernel matrix W -> (N, N)
    W = np.exp(-np.square(D) / epsilon)
    
    # form diagonal normalization matrix P -> (N, N)
    P = np.diag(W.sum(axis=1))

    # form kernel matrix K -> (N, N)
    P_inv = np.linalg.inv(P)
    K = P_inv @ W @ P_inv

    # form diagonal normalization matrix Q -> (N, N)
    Q = np.diag(K.sum(axis=1))
        
    # form T_hat -> (N, N)
    Q_sqrt_inv = np.linalg.inv(sqrtm(Q))
    T_hat = Q_sqrt_inv @ K @ Q_sqrt_inv
    
    # find L+1 largest eigenvalues and eigenvectors of T_hat
    values, vectors = np.linalg.eig(T_hat)
    indices = values.argsort()[::-1]
    al = values[indices][:L+1]
    vl = vectors[:, indices][:, :L+1]

    eigenvalues = np.sqrt(al ** (1/epsilon)) # -> (L+1, )
    eigenvectors = (Q_sqrt_inv @ vl) # -> (N, L+1 )
    return eigenvectors * eigenvalues