import pandas as pd
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split


# 1 Here we get the visualization of our data,
# remove outlayers and do sampling.
# This part only need to do once, we store the output to the 
# data file to make sure we use the same data.


def preprocessing(train_X,train_y):


	X_train, X_test, y_train, y_test = train_test_split(train_X,train_y)
	# if SMOTE:
	# 	sm = SMOTE(random_state = 42)
	# 	train_X, train_y = sm.fit_sample(train_X, train_y )
	# 	return X_train, X_test, y_train, y_test

	return X_train, X_test, y_train, y_test



