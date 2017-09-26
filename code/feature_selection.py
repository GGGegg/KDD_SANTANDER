import pandas as pd

# 3 Here we get the result from feature engineering 
# and reduce the dimension




# First we do the data clean work
# Step1: Remove constant variables : The dataset has many variables that are constant and has no significance with the customer satisfaction. These variables are identified and removed.

dim=train.shape[1]; # There's total 371-1=370 dims(-1 is the target 1)
sample=train.shape[0];
head=train.columns
train_datadim = head.values
train_head_list=train_datadim.tolist()#list

trainclear=train

for i in range(dim):
    dimlist=np.unique(train[train_head_list[i]])
    if len(dimlist)==1:
        trainclear= train.drop(train_head_list[i],axis=1,inplace=True)

print(trainclear.shape) # Now only has 336 features and 1 label
print(trainclear)
