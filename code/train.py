from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import scipy as sp
from imblearn.over_sampling import SMOTE 
from scipy.sparse import csr_matrix
import sys
import importlib
import log as log
import time


# 3 Here we built multiple classifiers
datasets = {
	"train":"data/train.csv",
	"test":"data/test.csv"
}

def aoc(act,pred):
	#this part is for offline, if we want to evaluate our performance by our own
	# we will define the aoc function here
	from sklearn.metrics import roc_auc_score
	result = roc_auc_score(act,pred)
	return result


def train(conf):

	tbegin = time.time()
	configures = importlib.import_module("conf."+conf)
	logger = log.Log("output/"+conf+"_log.csv")

	#feature engineering
	logger.log(configures.message)
	logger.log("feature engineering ...")
	train_X,train_y,true_test = configures.feature_engineering(datasets)

	tfe_end = time.time()
	logger.log("time: %.6f" % (time.time()-tbegin))

	#feature selection
	logger.log("feature_selection...")
	train_X = configures.feature_selection(train_X)
	train_X = configures.feature_selection_dr(train_X)
	true_test = configures.feature_selection(true_test)

	tfs_end = time.time()
	logger.log("time: %.6f" % (time.time()-tfe_end))

	#preprocessing
	logger.log("prepropressing...")
	X_train,X_test,y_train,y_test = configures.preprocessing(train_X,train_y)

	tprep_end = time.time()
	logger.log("time: %.6f" % (time.time()-tfe_end))


	#fitting/classification
	logger.log("fitting...")
	model = configures.model(X_train,y_train)
	tfit_end = time.time()
	logger.log("time: %.6f" % (time.time()-tfe_end))

	#result to log
	scores = model.predict_proba(X_test)
	loss = aoc(y_test, scores[:,1])
	logger.log("aoc:%.6f" % loss)
	scores = pd.DataFrame(scores)
	scores.to_csv("output/"+conf+"_scores.csv",header=None,index=None)
	logger.log("All time: %.6f" % (time.time()-tbegin))
	logger.log("------------------------------------------")
	logger.close()

if __name__ == "__main__":
	conf = sys.argv[1]
	train(conf)