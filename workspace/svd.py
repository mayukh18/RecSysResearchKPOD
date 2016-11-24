import numpy as np
from math import sqrt

def svd(utilMat, k=15):

    item_means=np.mean(utilMat, axis=0)
    item_means_tiled = np.tile(item_means, (utilMat.shape[0],1))

    utilMat = utilMat - item_means_tiled
    del item_means_tiled
    print(utilMat)

    U, s, V = np.linalg.svd(utilMat, full_matrices=False)

    U = U[:,0:k]
    V = V[0:k,:]
    s_root = np.diag([sqrt(s[i]) for i in range(0,k)])

    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)

    del utilMat, U, V, s_root

    UsV = np.dot(Usk, skV)
    print(UsV)

    for i in range(UsV.shape[0]):
        UsV[i,:] = UsV[i,:] + item_means

    return UsV