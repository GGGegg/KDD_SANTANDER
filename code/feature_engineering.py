import pandas as pd
from sklearn.ensemble import IsolationForest

# 2 Here we get the result from preprocessing and try to generate more features
def remove_outlier(data):
    pass

#generating outlier feature
def feature_engineering(datasets):
    #generate some features here, need discuss

    #read the file
    train = pd.read_csv(datasets["train"])
    train_X = train.iloc[:,:-1]
    train_y = train.TARGET

    # add outlier feature
    test= pd.read_csv(datasets["test"])
    n = train_X.shape[0]
    clf = IsolationForest(random_state=42)
    clf.fit(train_X)
    outlier = clf.predict(train_X)
    train_X = pd.DataFrame(train_X)
    train_X["outlier"] = outlier

    #add xxx

    return train_X,train_y,test

