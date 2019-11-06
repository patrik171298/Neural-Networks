import numpy as np
import pandas as pd
import matplotlib as plt
import argparse
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
from scipy import stats

def sigmoid(x):
	# Returns the sigmoid value transformation for any input X
	return 1/(1 + np.exp(-x))

def prediction(attributes, weights):
	# Calculates the predicted values and returns the sigmoid values
	predicted = np.sum(weights.T*attributes, axis=1)
	return sigmoid(predicted)

def gradient(X, weights, y):
	#Calculates the gradient based on the newly predicted values
	difference = y - prediction(X, weights)
	del_E = - np.sum(np.multiply(X.T, difference), axis=1)
	return del_E

def update(X, weights, y, learning_parameter):
	#updates the weights using the gradient descent algorithm
	updated_weights = weights - learning_parameter*gradient(X, weights, y)
	return updated_weights

def OnevsAll(X_train, X_test, y, y_actual, learning_parameter, num_classes, split):
    new_models=[[] for i in range(num_classes)]

    for i in range(len(new_models)):
        new_models[i]=np.where(y==i, 1, 0)

    predictions=[[] for i in range(num_classes)]
    probabilities=[[] for i in range(num_classes)]
    learning_parameter = 0.01
    np.random.seed(0)

    y_pred = []

    for pred in range(len(predictions)):
        weights = np.random.rand(7)
        y_train=new_models[pred][:split]
        y_test=new_models[pred][split:]

        for i in range(500):
            weights = update(X_train, weights, y_train, learning_parameter)

        probabilities[pred] = sigmoid(np.sum(np.multiply(X_test, weights), axis=1))
        predictions[pred] = np.heaviside((probabilities[pred]-0.5),0)
        print('Individual accuracy of class', pred, ' = ',accuracy_score(y_test, predictions[pred]), '\n')

    y_pred=np.argmax(probabilities, 0)
    print(confusion_matrix(y_actual, y_pred))
    print('Overall accuracy = ',accuracy_score(y_actual, y_pred))

def OnevsOne(X_train, X_test, y, y_actual, learning_parameter, num_classes, split):
    
    labels=np.unique(y).astype('str')
    new_models=[[] for i in range(int(num_classes*(num_classes-1)/2))]
    binary_class_models=[[] for i in range(int(num_classes*(num_classes-1)/2))]
    binary_class_labels=[[] for i in range(int(num_classes*(num_classes-1)/2))]

    for i in range(len(new_models)):
            new_models[i]=np.where(y==i+1, 1, 0)
            
    tempargs = [[] for i in range(int(num_classes*(num_classes-1)/2))]
    
    i=0
    for p in range(1, num_classes):
        for q in range(p):
            binary_class_labels[i]=labels[q]+labels[p]
            binary_class_models[i]=np.vstack((new_models[q], new_models[p]))
            i+=1
            
    for m, model in enumerate(binary_class_models):
        for arg in range(model.shape[1]):
            if model[0][arg] == model[1][arg]:
                tempargs[m].append(arg)

    binary_class_models=[model[1] for model in binary_class_models]

    new_models=binary_class_models
    predictions=[[] for i in range(num_classes)]
    probabilities=[[] for i in range(num_classes)]
    class_predictions=[[] for i in range(num_classes)]

    y_pred = []

    for pred in range(len(predictions)):
        print("Binary Class: ", binary_class_labels[pred])
        weights = np.random.rand(7)
        
        
        y_train=new_models[pred][:split]
        y_test=new_models[pred][split:]

        for i in range(1000):
            weights = update(X_train, weights, y_train, learning_parameter)

        class_labels = list(map(int, binary_class_labels[pred]))
        probabilities[pred] = sigmoid(np.sum(np.multiply(X_test, weights), axis=1))
        predictions[pred] = np.heaviside((probabilities[pred]-0.5),0).astype(int)
        print("Accuracy for Class",binary_class_labels[pred],": ", accuracy_score(y_test, predictions[pred]),"\n")
        class_predictions[pred]=[class_labels[label] for label in predictions[pred]]

    y_pred=stats.mode(class_predictions)[0][-1]
    print("Overall Accuracy: ", accuracy_score(y_actual, y_pred))
    print(confusion_matrix(y_actual, y_pred))

def main():

    data=pd.read_excel('data4.xlsx').sample(frac=1).reset_index(drop=True)
    dataset=pd.read_excel('data4.xlsx').sample(frac=1).reset_index(drop=True).values
    X=dataset[:,:7]
    y=(dataset[:,7]).astype(int)
    X_normalized = X/np.max(X, 0)

    split=int(np.round(0.6*X.shape[0]))
    X_train=X_normalized[:split,:]
    X_test=X_normalized[split:]

    y_actual=y[split:]
    learning_parameter=0.01

    num_classes=np.unique(y).shape[0]

    print('Please input Multiclass Logistic Regression Algorithm:\n'
          '1. One vs All Algorithm\n'
          '2. One vs One Algorithm\n')
    algo=int(input("Enter Algorithm: "))

    if algo == 1:
        OnevsAll(X_train, X_test, y-1, y_actual-1, learning_parameter, num_classes, split)
    elif algo == 2:
        OnevsOne(X_train, X_test, y, y_actual, learning_parameter, num_classes, split)

if __name__=="__main__":
    main()
