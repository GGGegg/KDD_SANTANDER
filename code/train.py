import pandas as pd
import sys
import time
import log as log

# 3 Here we built multiple classifiers
datasets = {
	"train":"data/train.csv",
	"test":"data/test.csv"
}

def aoc(act,pred):
	#this part is for offline, if we want to evaluate our performance by our own
	# we will define the aoc function here
	pass


def train(conf):

	t.begin = time.time()
	configures = importlib.import_module("conf."+conf)
	logger = log.Log("output/"+conf+"_log.csv")

	#feature engineering
	logger.log(configures.message)
	logger.log("feature engineering ...")
	X,y = configures.feature_engineering(datasets)
	tfe_end = time.time()
	logger.log("time: %.6f" % (time.time()-tbegin))

	#preprocessing
	logger.log("prepropressing...")
    X_train,X_test,y_train,y_test = configures.preprocessing(X,y)
    tprep_end = time.time()
    logger.log("time: %.6f" % (time.time()-tfe_end))

	#fitting/classification
	logger.log("fitting...")  
    model = configures.model(X_train,y_train)
    tfit_end = time.time()
    logger.log("time: %.6f" % (time.time()-tfe_end))

	#result to log
	scores = model.predict_proba(X_test)
    loss = aco(y_test, scores[:,1])
    logger.log("aoc:%.6f" % loss)
    scores = pd.DataFrame(scores)
    scores.to_csv("output/"+conf+"_scores.csv",header=None,index=None)
    logger.log("All time: %.6f" % (time.time()-tbegin))
    logger.log("------------------------------------------")
    logger.close()

if __name__ == "__main__":
	conf = sys.argv[1]
	train(conf)