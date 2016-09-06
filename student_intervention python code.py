# Import libraries
import numpy as np
import pandas as pd

from sklearn import cross_validation

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns

n_students = student_data.passed.size
n_features = student_data.columns.size -1
n_passed = student_data[(student_data.passed == 'yes')].size
n_failed = student_data[(student_data.passed == 'no')].size
grad_rate = (float(n_passed) / (float(n_passed) + float(n_failed)))
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate*100)

# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty
    
    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int
        
        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'
        
        outX = outX.join(col_data)  # collect column(s) in output dataframe
    
    return outX

X_all = preprocess_features(X_all)
#y_all = y_all.replace(['yes', 'no'], [1, 0])
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))


num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train


# --------------- Set up helper functions ----------------

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.24)

print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])

# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

# Train and predict using different training set sizes using predict_labels function
def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {}".format(predict_labels(clf, X_test, y_test))

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.3f}".format(end - start
    

# ------------------------------- Train all models ---------------------------------
# using above helper functions

import time
from sklearn import grid_search
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.24)

sizes = np.round(np.linspace(1, len(X_train), 4))[1:4]
sizes[0] = sizes[0]-1

train_err = np.zeros(len(sizes))
test_err = np.zeros(len(sizes))
times = np.zeros(len(sizes))

#parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
#reg = grid_search.GridSearchCV(regressor, parameters, scoring = 'mean_squared_error')


for i, s in enumerate(sizes): # enumerate through desired learning sizes
    s = int(s)
    # Create and fit the decision tree regressor model
    tree = DecisionTreeClassifier(max_depth = 5) #train/test multiple different depth assumptions
    tree2 = DecisionTreeClassifier(max_depth = 8) #train/test multiple different depth assumptions
    tree3 = DecisionTreeClassifier(max_depth = 1)
    svm = SVC(kernel = 'rbf')
    svm2 = SVC(kernel = 'linear')
    gauss  = GaussianNB()
    models = [tree, tree2, tree3, svm, svm2, gauss]
    
    X_train_sample = X_train[:s]
    Y_train_sample = y_train[:s]
    for model in models:
        print "\n Results for model {} \n".format(model)
        
        if model == 'svm' or model =='svm2':# scale data for Support Vector Machines
            X_train_sample = preprocessing.scale(X_train_sample)
            X_test_scaled = preprocessing.scale(X_test)
            train_classifier(model, X_train_sample, Y_train_sample)
            train_predict(model, X_train_sample, Y_train_sample, X_test_scaled, y_test)
        else:
            train_classifier(model, X_train_sample, Y_train_sample)
            train_predict(model, X_train_sample, Y_train_sample, X_test, y_test)

# Fit model to training data
#train_classifier(clf, X_train, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it

f1_scorer = make_scorer(f1_score, pos_label="yes")
classifier = DecisionTreeClassifier()
parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
grid_search = grid_search.GridSearchCV(classifier, parameters, scoring=f1_scorer)

grid_search.fit(X_train, y_train)
best_reg = grid_search.best_estimator_
print best_reg

train_predict(best_reg, X_train, y_train, X_test, y_test)









