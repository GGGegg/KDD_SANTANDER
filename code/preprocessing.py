import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor

# 1 Here we get the visualization of our data,
# remove outliers and do sampling.
# This part only need to do once, we store the output to the 
# data file to make sure we use the same data.


def preprocessing(train_X,train_y,SMOTE=False):
    train_X = normalize(train_X,axis=0)
    if SMOTE == True:
        sm = SMOTE(random_state=42)
        train_X,train_y = sm.fit_sample(train_X,train_y)
    X_train,X_test,y_train,y_test = train_test_split(train_X,train_y,random_state=42)
    return X_train,X_test,y_train,y_test





