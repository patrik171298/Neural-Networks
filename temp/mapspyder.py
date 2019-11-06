import numpy as np
import pandas as pd
import matplotlib as plt
import argparse
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix

def p_x_given_yk(x_vector, classes, label):
    Xk = classes[label]
    n = Xk.shape[0]
    mu = np.mean(Xk, axis = 0)
    cov_matrix = np.cov(Xk.T)
    K = 1/(np.sqrt(((2*np.pi)**n)*np.linalg.det(cov_matrix)))
    prob = K*np.exp(-0.5*np.linalg.multi_dot([((x_vector-mu).T), np.linalg.inv(cov_matrix), (x_vector-mu)]))
    return prob

def p_y(y, classes, label):
    return classes[label].shape[0]/y.shape[0]

def p_y_given_x(x_vector, y, classes, label):
    return p_x_given_yk(x_vector, classes, label) * p_y(y, classes, label)
    
def main():
    data = pd.read_excel('data4.xlsx')
    dataset = pd.read_excel('data4.xlsx', header = None).sample(frac = 1).reset_index(drop = True).values

    X = dataset[:,:7]
    y = dataset[:,7]-1
    #X_normalized=(X-X.mean(0))/X[:,0].std(0)
    #X = X_normalized
    #X = (X-X.mean(0))/X[:,0].std(0)
    split = int(np.round(0.7*X.shape[0]))
    X_train = X[:split,:]
    X_test = X[split:,:]
    y_train = y[:split]
    y_test = y[split:]
    
    num_classes = np.unique(y).shape[0]
    classes = [[] for i in range(num_classes)]
    
    
    for i in range(X_train.shape[0]):
        classes[int(y_train[i])].append(X_train[i])
    
    classes = [np.stack(label) for label in classes]
    y_pred = []

    for j in range(X_test.shape[0]):
        predictions = []
        for label in range(num_classes):
            predictions.append(p_y_given_x(X_test[j], y_train, classes, label))
        yk = int(np.argmax(predictions))
        y_pred.append(yk)

    y_pred = np.array(y_pred)
    
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
if __name__=="__main__":
    main()