import pandas as pd

# 2 Here we get the result from preprocessing and try to generate more features

def feature_engineering(datasets):
	#generate some features here, need discuss
	
	train = pd.read_csv(datasets["train"])
	train_X = train.iloc[:,:-1]
	train_y = train.TARGET
	test= pd.read_csv(datasets["test"])
	return train_X,train_y,test