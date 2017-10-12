# Here we give our parameters. eg. which classifier you will use.
# Each one create there own file, so we will not missed up.

import sys
sys.path.append("..code")
import pandas as pd 
import code.preprocessing as ppo
import code.feature_engineering as fe
# import feature_selection as fs 


# about online and offline, pending to discussion
mode = "online"
mode = "offline"
message="baseline"
SMOTE="False"
feature_engineering = fe.feature_engineering
preprocessing = ppo.preprocessing
# feature_selection = fs.feature_selection
