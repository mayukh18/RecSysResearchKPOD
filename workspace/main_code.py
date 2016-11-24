import numpy as np
import pandas as pd
from metrics import rmse
from svd import svd
from kpod import kpod
from methods import create_utility_matrix


data = pd.read_csv("../datasets/movielens1m.csv")
#print data

data['userId'] = data['userId'].astype('string')
data['movieId'] = data['itemId'].astype('string')

data = data.iloc[np.random.permutation(len(data))]
data.reset_index(inplace=True)

sets = []
rowlen = len(data)
for i in range(5):
    sets.append(data[int(i*rowlen/5):int((i+1)*rowlen/5)])

errors = []

for i in range(5):

    train_ids = [ sets[k] for k in range(5) if k!=i ]
    train = pd.concat(train_ids).reset_index()
    test = sets[i].reset_index()

    user_item_matrix = create_utility_matrix(train, formatizer={'user':'userId', 'item':'movieId', 'value':'rating'})

    users = list(user_item_matrix.index)
    items = list(user_item_matrix.columns)

    print("no. of users: ",len(users))
    print("no. of items: ",len(items))


    user_index = {users[i]: i for i in range(len(users))}
    item_index = {items[i]: i for i in range(len(items))}

    del users
    del items

    mask = np.isnan(user_item_matrix)
    masked_arr = np.ma.masked_array(user_item_matrix, mask)

    del mask
    del user_item_matrix

    item_means=np.mean(masked_arr, axis=0)
    #user_means=np.mean(masked_arr, axis=1)
    #item_means_tiled = np.tile(item_means, (user_item_matrix.shape[0],1))
    #init_dgesdd failed init

    print masked_arr
    utilMat = masked_arr.filled(item_means)
    print(utilMat)

    #utilMat = kpod(utilMat=utilMat, mask=masked_arr, iter=40, n_clusters=10, method="normal")
    utilMat = svd(utilMat,k=15)

    pred = [] #to store the predicted ratings

    for _,row in test.iterrows():
        user = row['userId']
        item = row['movieId']

        if user in user_index:
            u_index = user_index[user]
            if item in item_index:
                i_index = item_index[item]
                pred_rating = utilMat[u_index, i_index]
            else:
                pred_rating = np.mean(utilMat[u_index, :])
        else:
            if item in item_index:
                i_index = item_index[item]
                pred_rating = np.mean(utilMat[:, i_index])
            else:
                pred_rating = np.mean(utilMat[:, :])

        pred.append(pred_rating)

    error = rmse(test['rating'], pred)
    print(error)

    errors.append(error)
    del error, pred

print np.mean(errors)