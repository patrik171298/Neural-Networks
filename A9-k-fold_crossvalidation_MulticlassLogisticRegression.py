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
	predicted = np.sum(weights.T*attributes, axis = 1)
	return sigmoid(predicted)

def gradient(X, weights, y):
	#Calculates the gradient based on the newly predicted values
	difference = y - prediction(X, weights)
	del_E = - np.sum(np.multiply(X.T, difference), axis = 1)
	return del_E

def update(X, weights, y, learning_parameter):
	#updates the weights using the gradient descent algorithm
	updated_weights = weights - learning_parameter*gradient(X, weights, y)
	return updated_weights

def OnevsAll(dataset, temp_indices, train_dataset, test_dataset, learning_parameter, num_classes, weights):

    y = (dataset[:,7]-1).astype(int)
    new_models = [[] for i in range(num_classes)]

    X_train = train_dataset[:,:7]/ np.max(train_dataset[:,:7], 0)
    y_train = train_dataset[:,7]-1
    X_test = test_dataset[:,:7]/ np.max(test_dataset[:,:7], 0)
    y_actual = test_dataset[:,7]-1

    for i in range(len(new_models)):
        new_models[i] = np.where(y == i, 1, 0)

    predictions = [[] for i in range(num_classes)]
    probabilities = [[] for i in range(num_classes)]
    learning_parameter = 0.01
    np.random.seed(0)

    y_pred = []

    for pred in range(len(predictions)):


        y_test = new_models[pred][temp_indices]
        y_train = np.delete(new_models[pred], temp_indices, axis = 0)

        for i in range(500):
            weights = update(X_train, weights, y_train, learning_parameter)

        probabilities[pred] = sigmoid(np.sum(np.multiply(X_test, weights), axis = 1))
        predictions[pred] = np.heaviside((probabilities[pred]-0.5),0)

    y_pred = np.argmax(probabilities, 0)
    print(confusion_matrix(y_actual, y_pred))
    print('Individial accuracy = ',accuracy_score(y_actual, y_pred),'\n')
    acc_score = accuracy_score(y_actual, y_pred)*len(temp_indices)

    return weights, acc_score

def OnevsOne(dataset, temp_indices, train_dataset, test_dataset, learning_parameter, num_classes, weights):

    y = (dataset[:,7]).astype(int)
    new_models = [[] for i in range(num_classes)]

    X_train = train_dataset[:,:7]/ np.max(train_dataset[:,:7], 0)
    y_train = train_dataset[:,7].astype(int)
    X_test = test_dataset[:,:7]/ np.max(test_dataset[:,:7], 0)
    y_actual = test_dataset[:,7].astype(int)

    labels = np.unique(y).astype('str')
    new_models = [[] for i in range(int(num_classes*(num_classes-1)/2))]
    binary_class_models = [[] for i in range(int(num_classes*(num_classes-1)/2))]
    binary_class_labels = [[] for i in range(int(num_classes*(num_classes-1)/2))]

    for i in range(len(new_models)):
            new_models[i] = np.where(y == i+1, 1, 0)

    i = 0
    for p in range(1, num_classes):
        for q in range(p):
            binary_class_labels[i] = labels[q]+labels[p]
            binary_class_models[i] = np.vstack((new_models[q], new_models[p]))
            i += 1

    binary_class_models = [model[1] for model in binary_class_models]

    new_models = binary_class_models
    predictions = [[] for i in range(num_classes)]
    probabilities = [[] for i in range(num_classes)]
    class_predictions = [[] for i in range(num_classes)]

    y_pred = []

    for pred in range(len(predictions)):

        y_test = new_models[pred][temp_indices]
        y_train = np.delete(new_models[pred], temp_indices, axis = 0)

        weights = np.random.rand(7)

        for i in range(5000):
            weights = update(X_train, weights, y_train, learning_parameter)

        class_labels = list(map(int, binary_class_labels[pred]))
        probabilities[pred] = sigmoid(np.sum(np.multiply(X_test, weights), axis = 1))
        predictions[pred] = np.heaviside((probabilities[pred]-0.5),0).astype(int)
        class_predictions[pred] = [class_labels[label] for label in predictions[pred]]

    y_pred = stats.mode(class_predictions)[0][-1]
    print("Individual Accuracy: ", accuracy_score(y_actual, y_pred),'\n')
    acc_score = accuracy_score(y_actual, y_pred)*len(temp_indices)

    return weights, acc_score

def main():

    data = pd.read_excel('data4.xlsx', header = None).sample(frac = 1).reset_index(drop = True)
    dataset = pd.read_excel('data4.xlsx').sample(frac = 1).reset_index(drop = True).values
    X = dataset[:,:7]
    y = (dataset[:,7]).astype(int)

    num_classes = np.unique(y).shape[0]

    k = int(input("Enter k: "))

    learning_parameter = 0.01
    batch = np.round(dataset.shape[0]/k).astype(int)
    temp_dataset = dataset
    split_point = 0
    weights = np.random.rand(7)
    overall_accuracy = 0
    batch_number = 1

    print('Please input Multiclass Logistic Regression Algorithm:\n'
          '1. One vs All Algorithm\n'
          '2. One vs One Algorithm\n')
    algo = int(input("Enter Algorithm: "))

    while (split_point+batch)<dataset.shape[0]:
        print("\nBatch ", batch_number)
        batch_number += 1
        temp_indices = range(split_point, split_point+batch)
        test_dataset = dataset[temp_indices]
        train_dataset = np.delete(temp_dataset, temp_indices, axis = 0)

        if algo == 1:
            weights, acc_score = OnevsAll(dataset, temp_indices, train_dataset, test_dataset, learning_parameter, num_classes, weights)
        elif algo == 2:
            weights, acc_score = OnevsOne(dataset, temp_indices, train_dataset, test_dataset, learning_parameter, num_classes, weights)

        split_point = split_point+batch
        overall_accuracy += acc_score


    else:
        print("\nBatch ", batch_number)
        batch_number += 1
        temp_indices = range(split_point, dataset.shape[0])
        test_dataset = dataset[temp_indices]
        train_dataset = np.delete(temp_dataset, temp_indices, axis = 0)

        if algo == 1:
            weights, acc_score = OnevsAll(dataset, temp_indices, train_dataset, test_dataset, learning_parameter, num_classes, weights)
        elif algo == 2:
            weights, acc_score = OnevsOne(dataset, temp_indices, train_dataset, test_dataset, learning_parameter, num_classes, weights)

        split_point = split_point+batch
        overall_accuracy += acc_score

    overall_accuracy = overall_accuracy/dataset.shape[0]
    print("Overall Accuracy = ",overall_accuracy)

if __name__=="__main__":
    main()
