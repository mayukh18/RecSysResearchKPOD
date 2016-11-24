import numpy as np
import pandas as pd



def create_utility_matrix(data, formatizer = {'user':0, 'item': 1, 'value': 2}):

    itemField = formatizer['item']
    userField = formatizer['user']
    valueField = formatizer['value']

    userList = data.ix[:,userField].tolist()
    itemList = data.ix[:,itemField].tolist()
    valueList = data.ix[:,valueField].tolist()

    users = list(set(data.ix[:,userField]))
    items = list(set(data.ix[:,itemField]))

    users_index = {users[i]: i for i in range(len(users))}



    pd_dict = {item: [np.nan for i in range(len(users))] for item in items}

    for i in range(0,len(data)):
        item = itemList[i]
        user = userList[i]
        value = valueList[i]

        pd_dict[item][users_index[user]] = value
        #print i

    X = pd.DataFrame(pd_dict)
    X.index = users

    return X
