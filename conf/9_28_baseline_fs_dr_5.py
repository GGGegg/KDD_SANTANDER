# Here we give our parameters. eg. which classifier you will use.
# Each one create there own file, so we will not missed up.

#import sys
#sys.path.append("code")
import pandas as pd 
import code.preprocessing as ppo
import code.feature_engineering as fe
import code.feature_selection as fs
import code.classifier as cf

# about online and offline, pending to discussion
mode = "online"
mode = "offline"
message="baseline"
component = 5
SMOTE="False"
feature_engineering = fe.feature_engineering
preprocessing = ppo.preprocessing
feature_selection = fs.remove_identical_features
feature_selection_dr = fs.feature_representation_PCA
model = cf.logistic_regression_classifier
# model = cf.MLP_classifier