import pandas as pd
import numpy as np
import itertools
from sklearn.feature_selection import VarianceThreshold
# from sklearn.preprocessing import normalization

# 3 Here we get the result from feature engineering 
# and reduce the dimension




# First we do the data clean work
# Step1: Remove constant variables : The dataset has many variables that are constant and has
# no significance with the customer satisfaction. These variables are identified and removed.

# dim=train.shape[1]; # There's total 371-1=370 dims(-1 is the target 1)
# sample=train.shape[0];
# head=train.columns
# train_datadim = head.values
# train_head_list=train_datadim.tolist()#list

# trainclear=train

# for i in range(dim):
#     dimlist=np.unique(train[train_head_list[i]])
#     if len(dimlist)==1:
#         trainclear= train.drop(train_head_list[i],axis=1,inplace=True)

# print(trainclear.shape) # Now only has 336 features and 1 label
# print(trainclear)

def remove_identical_features(data,target=None):
    print("Remove identical features")
    print("original data shape: ",data.shape)
    for feature_1,feature_2 in itertools.combinations(
                iterable = data.columns, r =2):
        if np.array_equal(data[feature_1],data[feature_2]):
            data.drop(feature_2,axis = 1)
    print("distinct data shape:", data.shape)
    return data

def feature_representation_PCA(data,target=None,component = 40):
    print("PCA...")
    from sklearn.decomposition import PCA
    component = int(data.shape[1]/2)
    pca = PCA(n_components = component)
    # outlier = data.outlier
    # data_rest = data.drop(['outlier'])
    transformed_data = pca.fit_transform(data)
    pca_attrs = pd.DataFrame()
    pca_attrs[0] = pca.explained_variance_
    pca_attrs[1] = pca.explained_variance_ratio_
    pca_attrs.columns = ["pca.explained_variance_","pca.explained_variance_ratio_"]
    pca_attrs.to_csv("output/pca_attr.csv",index=None)
    # transformed_data = pd.concat(transformed_data,data_rest)
    print(transformed_data.shape)
    return transformed_data

def feature_representation_LLE(data,target=None,component = 168):
    from sklearn.manifold import LocallyLinearEmbedding
    lle = LocallyLinearEmbedding(n_neighbors = 5, n_components = component,
                                eigen_solver = 'dense', method = 'standard')
    transformed_data = lle.fit_transform(data)
    lle_error = pd.DataFrame()
    lle_error[0] = lle.reconstruction_error
    lle_error.columns = ["lle.reconstruction_error"]
    lle_error.to_csv("output/lle_error.csv",index = None)
    return transformed_data

def select_base_importance(data,target):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    clf = ExtraTreesClassifier(random_state=42)
    clf.fit(data,target)
    importance = pd.Series(clf.feature_importances_,index=data.columns.values).sort_values(ascending=False)
    # print(importance)
    imp = pd.DataFrame()
    imp["id"] = importance.index
    imp["importance"]= importance.values
    imp.to_csv("output/feature_importance.csv",index=None)
    model = SelectFromModel(clf,prefit=True)
    X_new = model.transform(data)
    print(X_new.shape)
    return X_new
