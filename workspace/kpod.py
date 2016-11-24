from sklearn.cluster import KMeans
import numpy as np



def kpod(utilMat, mask, iter=40, n_clusters=3, method='transpose'):

    if method=='transpose':
        utilMat = np.transpose(utilMat)
        mask = np.transpose(mask)

    print mask
    print utilMat

    for ii in range(iter):
        print ii
        cls = KMeans(n_clusters=n_clusters, n_init=6)
        labels_hat = cls.fit_predict(utilMat)

        #utilMat = mask.filled(cls.cluster_centers_[labels_hat])

        for jj in range(len(labels_hat)):
            c = labels_hat[jj]
            utilMat[jj, :] = mask[jj, :].filled(cls.cluster_centers_[c])


    cluster_indices = [ [] for i in range(n_clusters) ]
    for i in range(len(labels_hat)):
        cluster_indices[labels_hat[i]].append(i)

    for i in range(n_clusters):
        print len(cluster_indices[i])

    if method=='transpose':
        utilMat = np.transpose(utilMat)

    return utilMat