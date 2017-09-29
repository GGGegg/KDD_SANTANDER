import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
# if your model uses random_state, please assign this value.
seed = 42
# 3 Here we built multiple classifiers

# We use this one for test other techniques, because it is fast.
def logistic_regression_classifier(train_X,train_y):
	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression(penalty='l2',n_jobs=20,solver="sag")
	model.fit(train_X, train_y) 
	return model

      
     
 
def MLP_classifier(train_X,train_y):
	from sklearn.neural_network import MLPClassifier
	scaler = preprocessing.MinMaxScaler(copy=False)
	train_X = scaler.fit_transform(train_X)
	mlp_classifier = MLPClassifier(random_state = seed)
	print(mlp_classifier.get_params().keys())
	parameters = {
				"hidden_layer_sizes":[370,185,92,46,23,11],
				"activation":['identity','logistic','relu'],
				"solver":["sgd"],
				"learning_rate":["adaptive"],
				"max_iter":[500],
				"random_state":[seed],
				"learning_rate_init":[0.001,0.01,0.02,0.03,0.1]
	}

	classifier = GridSearchCV(mlp_classifier,param_grid = parameters, pre_dispatch = 5,
									cv = 10,n_jobs=1,error_score = 0)
	classifier.fit(train_X,train_y)
	return classifier
 def logistic_regression_classifier(train_X,train_y):	
	from sklearn.linear_model import LogisticRegression
	from sklearn.neighbors import KNeighborsClassifier
	neigh = KNeighborsClassifier(n_neighbors=7)
	neigh.fit(train_X,train_y) 
	return classifier

