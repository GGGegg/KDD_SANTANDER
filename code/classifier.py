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

def random_forest(train_X,train_y):
    from sklearn.ensemble import RandomForestClassifier
    scaler = preprocessing.MinMaxScaler(copy=False)
    train_X = scaler.fit_transform(train_X)
    model = RandomForestClassifier()
    # parameters = {"n_estimators":[10,20,30],
    #               "max_features":[15,20],
    #               "max_depth":[10,20,30],
    # }
    # classifier = GridSearchCV(model,param_grid=parameters,cv=10,error_score=0,pre_dispatch=5,n_jobs=1)
    # classifier.fit(train_X,train_y)
    # print(classifier.best_estimator_)
    model.fit(train_X,train_y)
    return model

def extra_tree_classsifier(train_X,train_y):
    from sklearn.ensemble import ExtraTreesClassifier
 
def MLP_classifier(train_X,train_y):
    from sklearn.neural_network import MLPClassifier
    # scaler = preprocessing.MinMaxScaler(copy=False)
    # train_X = scaler.fit_transform(train_X)
    mlp_classifier = MLPClassifier(random_state = seed)
    # print(mlp_classifier.get_params().keys())
    parameters = {
                # "hidden_layer_sizes":[370,185,92,46,23,11],
                # "activation":['identity','logistic','relu'],
                # "solver":["sgd"],
                # "learning_rate":["adaptive"],
                # "max_iter":[500],
                # "random_state":[seed],
                # "learning_rate_init":[0.001,0.01,0.02,0.03,0.1]
                "hidden_layer_sizes":[185,92],
                "activation":['relu'],
                "solver":["sgd"],
                "learning_rate":["adaptive"],
                "max_iter":[500],
                "random_state":[seed],
                "learning_rate_init":[0.001,0.01]
    }

    classifier = GridSearchCV(mlp_classifier,param_grid = parameters, pre_dispatch = 5,
                                    cv = 10,n_jobs=1,error_score = 0)
    classifier.fit(train_X,train_y)
    print(classifier.best_estimator_)
    return classifier

def KNN_classifier(train):
    
    df = train
    df.drop(['ID'], 1, inplace=True)
    
    X = np.array(df.drop(['TARGET'],1))
    y = np.array(df['TARGET'])
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    accuracy=clf.score(X_test, Y_test)
    print(accuracy)
    return clf
