import numpy as np
from scipy.linalg import sqrtm


class DiffusionMaps:
    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None

    def diffusionMaps(self, X, L):

        # create distance matrix D
        N = X.shape[0]
        D = np.empty((N, N)) 
        for i in range(N):
            for j in range(i, N):
                D[i, j] = np.linalg.norm(X[i] - X[j])
                D[j, i] = D[i, j]

        epsilon = 0.05 * D.max()

        # create W
        W = np.exp(-np.square(D) / epsilon)

        # create K
        P_inv = np.linalg.inv(np.diag(W.sum(axis=1)))
        K = P_inv @ W @ P_inv 

        # create T hat
        Q_inv_sqrt = np.linalg.inv(sqrtm(np.diag(K.sum(axis=1))))
        T_hat = Q_inv_sqrt @ K @ Q_inv_sqrt

        # get eigenvalues of T hat and sort them
        al, vl = np.linalg.eig(T_hat)
        indices = al.argsort()[::-1]
        al = al[indices][:L+1]  # (L+1,)
        vl = vl[:, indices][:, :L+1]  # (N, L+1)

        self.eigenvalues = np.sqrt(al ** (1 / epsilon))[1:]  # (L+1,) -> (L,)

        self.eigenvectors = (Q_inv_sqrt @ vl)[:, 1:]  # (N, N) x (N, L+1) = (N, L+1) -> (N, L)

        return self.eigenvectors * self.eigenvalues.reshape(1, -1)


        