import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from reco.recommender import SVDRecommender
from reco.datasets import load_movielens
from math import sqrt

data = load_movielens()

data['userId'] = data['userId'].astype('string')
data['movieId'] = data['movieId'].astype('string')

users = list(set(data['userId'])) #list of all users
movies = list(set(data['movieId'])) #list of all movies

test = pd.DataFrame(columns=data.columns)
train = pd.DataFrame(columns=data.columns)

test_ratio = 0.2 #adjust it as per wish

for u in users[:]:
    temp = data[data['userId'] == u]
    n = len(temp)
    test_size = int(test_ratio*n)

    temp = temp.sort_values('timestamp').reset_index()
    temp.drop('index', axis=1, inplace=True)

    dummy_test = temp.ix[n-1-test_size :]
    dummy_train = temp.ix[: n-2-test_size]

    test = pd.concat([test, dummy_test])
    train = pd.concat([train, dummy_train])


##################################################


svd = SVDRecommender(no_of_features=4)
user_item_matrix = svd.create_utility_matrix(train, formatizer={'user':'userId', 'item':'movieId', 'value':'rating'})


users = list(user_item_matrix.index)
items = list(user_item_matrix.columns)

user_index = {users[i]: i for i in range(len(users))}
item_index = {items[i]: i for i in range(len(items))}

####################################################

mask = np.isnan(user_item_matrix)
masked_arr = np.ma.masked_array(user_item_matrix, mask)



predMask = ~mask

item_means=np.mean(masked_arr, axis=0)
user_means=np.mean(masked_arr, axis=1)
item_means_tiled = np.tile(item_means, (user_item_matrix.shape[0],1))

utilMat = masked_arr.filled(item_means)
#utilMat = utilMat - item_means_tiled

utilMatTrans = np.transpose(utilMat)
masked_arr_Trans = np.transpose(masked_arr)
print masked_arr_Trans

print utilMat

n_clusters = 3

for ii in range(200):
    print ii
    cls = KMeans(n_clusters=n_clusters, n_init=12)
    labels_hat = cls.fit_predict(utilMatTrans)

    utilMatTrans = masked_arr_Trans.filled(cls.cluster_centers_[labels_hat])


#cluster_indices = [ [] for i in range(n_clusters) ]
#for i in range(len(labels_hat)):
#    cluster_indices[labels_hat[i]].append(i)

#for i in range(n_clusters):
#    print len(cluster_indices[i])


utilMat = np.transpose(utilMatTrans)
print utilMat

item_means=np.mean(utilMat, axis=0)
item_means_tiled = np.tile(item_means, (user_item_matrix.shape[0],1))

utilMat = utilMat - item_means_tiled


k = 15
U, s, V = np.linalg.svd(utilMat, full_matrices=False)

U = U[:,0:k]
V = V[0:k,:]
s_root = np.diag([sqrt(s[i]) for i in range(0,k)])

Usk=np.dot(U,s_root)
skV=np.dot(s_root,V)
UsV = np.dot(Usk, skV)

UsV = UsV + item_means_tiled

#######################################################

def rmse(true, pred):
    # this will be used towards the end
    x = true - pred
    return sum([xi*xi for xi in x])/len(x)

pred = [] #to store the predicted ratings

for _,row in test.iterrows():
    user = row['userId']
    item = row['movieId']

    u_index = user_index[user]
    if item in item_index:
        i_index = item_index[item]
        pred_rating = UsV[u_index, i_index]
    else:
        pred_rating = np.mean(UsV[u_index, :])
    pred.append(pred_rating)


print(rmse(test['rating'], pred))

"""


x = np.array([[1, 2, 3, 4, 5],[7,6,5,4,3],[11,11,11,12,21],[54,0,1,23,88]])
print x

print np.tile(x[:,1], (4,1))


ind = [0,2]
a = []
#a.append(x[0,:])
#a.append(x[2,:])
a = np.array(x[ind,:])

#print a
"""