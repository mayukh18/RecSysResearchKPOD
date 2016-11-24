def rmse(true, pred):
    # this will be used towards the end
    x = true - pred
    return sum([xi*xi for xi in x])/len(x)