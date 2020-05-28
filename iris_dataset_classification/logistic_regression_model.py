#logistic regression for iris dataset classification

import pandas as pd
import numpy as np 
import sklearn.model_selection as model_selection
import scipy.optimize as opt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

#input data
data = pd.read_csv('iris.csv', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
X = data.iloc[:, :data.shape[1]-1]
X.insert(0, 'ones', 1)
y = data['class']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 42)

'''
model = LogisticRegression()
model.fit(X_train, y_train)
predicted_classes = model.predict(X_test)
accuracy = accuracy_score(predicted_classes, y_test)
print('accuracy of model (using inbuilt methods) : {}'.format(accuracy))
'''

target = pd.get_dummies(y_train)

lbl_enc = LabelEncoder()
y_test = lbl_enc.fit_transform(y_test)
y_train = lbl_enc.fit_transform(y_train)

X_train = np.matrix(X_train)
X_test = np.matrix(X_test)
target = np.matrix(target) #data showing the one hot encoding of all the target classes
y_test = np.matrix(y_test).T

#logistic regression model
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def hyp(X, theta):
    z = X * theta
    h = sigmoid(z)
    return h

def scale_features(features):
    high = np.max(features)
    low = np.min(features)
    mean = np.mean(features)
    features = features - mean
    features /= (high - low)
    return features

def cost_func(theta, X, y, regParam):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    if theta.shape[0] < theta.shape[1]:
        theta = theta.T
    m = X.shape[0]
    h = hyp(X, theta)
    term1 = y.T * np.log(h)
    term2 = (1-y).T * np.log(1-h)
    regTerm = 1/2 * regParam * np.sum(np.power(theta,2))
    ans = 1/m * (-1 * (term1 + term2) + regTerm)
    return float(ans)

def gradient(theta, X, y, regParam):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    if theta.shape[0] < theta.shape[1]:
        theta = theta.T
    m = len(X)
    grad = np.matrix(np.zeros(theta.shape))
    error = (hyp(X, theta) - y)
    for i in range(theta.shape[0]):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i,:] = (1/m * np.sum(term))
        else:
            grad[i,:] = (1/m * np.sum(term)) + (regParam/m) * float(theta[i])
    return grad

alpha = 1 #learning rate
regParam = 1 #regularization parameter
#X_train = scale_features(X_train) -> Scaling has a negative effect on the output of the model
#X_test = scale_features(X_test)
num_classes = target.shape[1]
num_features = X.shape[1]
theta = np.matrix(np.zeros((num_features, num_classes))) #num of features x num_of_classes -> theta matrix for each of the target class

for class_ind in range(num_classes):
    init_cost = cost_func(theta[:, class_ind], X_train, target[:, class_ind], regParam)
    print('cost func value initially (for the class_ind {}) : {}'.format(class_ind, init_cost))
    result = opt.fmin_tnc(func = cost_func, x0 = theta[:, class_ind], fprime = gradient, args = (X_train, target[:, class_ind], regParam))
    if theta.shape == np.matrix(result[0]):
        theta[:, class_ind] = np.matrix(result[0])
    else:
        theta[:, class_ind] = np.matrix(result[0]).T
    final_cost = cost_func(theta[:, class_ind], X_train, target[:, class_ind], regParam)
    print('cost func final value (for the class_ind {}) : {}'.format(class_ind, final_cost))

#testing the model
predictions = np.full((X_test.shape[0], num_classes), -1.1)
final_predictions = np.full((X_test.shape[0], 1), -1.1)
for row_ind in range(X_test.shape[0]):
    for class_ind in range(num_classes):
        predictions[row_ind, class_ind] = hyp(X_test[row_ind,:], theta[:, class_ind])
    final_predictions[row_ind] = np.argmax(predictions[row_ind, :])
corrects = [1 if (prediction == output) else 0 for prediction, output in zip(final_predictions, y_test)]
accuracy = sum(corrects)/len(corrects)
print('accuracy of model : {}'.format(accuracy))
