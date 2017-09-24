import pandas as pd

# 2 Here we get the result from preprocessing and try to generate more features

def feature_engineering(datasets):
	#generate some features here, need discuss
	
	train = pd.read_csv(datasets["train"])
	test= pd.read_csv(datasets["test"])
	return train,test