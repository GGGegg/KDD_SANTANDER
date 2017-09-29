from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv("../data/train.csv")
train_X = train.iloc[:,:-1]
y = train.TARGET
y = y.reshape(len(y),1)

train_X = normalize(train_X)

pca = PCA(n_components = 2)
transformed_data = pca.fit_transform(train_X)
transformed_data = np.concatenate((transformed_data,y),axis=1)
plt.figure()
plt.title("PCA Visualization")
colors = ['blue','red']
target_name = ["satisfied","unsatisfied"]


for color,tar,name in zip(colors,[0,1],target_name):
	plt.scatter(transformed_data[ transformed_data[:,2] == tar,0],transformed_data[transformed_data[:,2] == tar,1],color=color, 
		alpha = 0.8, label = name, lw=2)
plt.legend(loc = "best",shadow=False,scatterpoints=1)
# plt.show()

