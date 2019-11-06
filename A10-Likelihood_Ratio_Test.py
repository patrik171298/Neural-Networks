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

def LRT(x_vector, y, classes):
    delta1 = p_x_given_yk(x_vector, classes, 0)/p_x_given_yk(x_vector, classes, 1)
    delta2 = p_y(y, classes, 1)/p_y(y, classes, 0)
    return delta1, delta2

def main():
    data = pd.read_excel('data3.xlsx', header = None)
    dataset = pd.read_excel('data3.xlsx', header = None).sample(frac = 1).reset_index(drop = True).values

    X = dataset[:,:4]
    y = dataset[:,4]-1
    split = int(np.round(0.6*X.shape[0]))
    X_train = X[:split,:]
    X_test = X[split:,:]
    y_train = y[:split]
    y_test = y[split:]

    num_classes = np.unique(y).shape[0]
    classes = [[] for i in range(num_classes)]


    for i in range(X_train.shape[0]):
        classes[int(y_train[i])].append(X_train[i])

    classes = [np.stack(label) for label in classes]
    print("Class lengths: ", classes[0].shape[0], ", ", classes[1].shape[0])
    y_pred = []

    for j in range(X_test.shape[0]):
        delta1, delta2 = LRT(X_test[j], y_train, classes)

        if delta1>delta2:
            y_pred.append(0)
        else:
            y_pred.append(1)

    y_pred = np.array(y_pred)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix")
    print(confusion_matrix(y_test, y_pred))

if __name__=="__main__":
    main()
