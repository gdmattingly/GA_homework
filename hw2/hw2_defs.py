from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def load_iris_data() :

    iris = datasets.load_iris()

    return (iris.data, iris.target, iris.target_names)

def knn(X_train, y_train, k_neighbors = 3 ) :
    # method returns a kNN object with methods:
    #     score(X_test, y_test) --> to score the model ising a test set
    #     predict(X_classify, y_test) --> to predict a result using the trained model

    clf = KNeighborsClassifier(k_neighbors)
    clf.fit(X_train, y_train)

    return clf

def cross_validate(XX, yy, classifier, k_fold, c_value) :

    k_fold_indices = KFold(len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=7)

    k_score_total=0
    for train_slice, test_slice in k_fold_indices :

        model=classifier(XX[[ train_slice ]],
                         yy[[ train_slice ]],c_value)

        k_score = model.score(XX[[ test_slice ]],
                              yy[[ test_slice ]])

        k_score_total += k_score

    return k_score_total*1.0/k_fold

def nb(X_train, y_train) :
    # this function returns a Naive Bayes object
    #  useful methods of this object for this exercise:
    #   fit(X_train, y_train) --> fit the model using a training set
    #   predict(X_classify) --> to predict a result using the trained model
    #   score(X_test, y_test) --> to score the model using a test set

    gnb = GaussianNB()
    clf = gnb.fit(X_train, y_train)

    return clf

def log_reg(X_train, y_train, c_value):

    # This line creates a LogisticRegression object clf where C=c_value
    clf = LogisticRegression(C=c_value)

    # This line fits the object clf to the data
    clf.fit(X_train,y_train)

    return clf

def get_c_values():
    #This function gets a range and step size for C values from the user
    #It then uses the numpy.arange method to return a numpy array of C values
    print 'Please input upper and lower range limits for C:'                                                    
    c_upper = float(raw_input('Please enter the largest C value:'))                                             
    c_lower = float(raw_input('Please input the smallest C value (must be >0):'))                                            
    c_interval = float(raw_input('Please input the increments of C to test within this range:'))               

    c_values = np.arange(c_lower,c_upper,c_interval,dtype=float) 
    return c_values
