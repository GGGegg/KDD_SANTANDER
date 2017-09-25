import pandas as pd

# 3 Here we built multiple classifiers

# We use this one for test other techniques, because it is fast.
def logistic_regression_classifier(train_X,train_y):
	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression(penalty='l2',n_jobs=20,solver="sag")
	model.fit(train_X, train_y) 
	return model

      
     
    


def MLP_classifier(train_X,train_y):
	pass

