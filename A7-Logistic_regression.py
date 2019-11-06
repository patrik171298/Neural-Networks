import numpy as np
import pandas as pd
import matplotlib as plt
import argparse
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix

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

def main():
	dataset=pd.read_excel('data3.xlsx').sample(frac=1).reset_index(drop=True).values

	X = dataset[:,:4]
	y = dataset[:,4]-1

	split = int(np.round(0.6*X.shape[0]))
	X_train = X[:split,:]
	y_train = y[:split]
	X_test = X[split:]
	y_test = y[split:]

	learning_parameter = 0.01
	np.random.seed(0)
	weights = np.random.rand(4)

	for i in range(15):
		#print(weights)
		weights = update(X_train, weights, y_train, learning_parameter)
		test_prediction = sigmoid(np.sum(np.multiply(X_test, weights), axis=1))
		y_pred = np.heaviside((test_prediction-0.5),0)
		print("Accuracy: ",accuracy_score(y_test, y_pred))

	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	print("Specificity: ", cm[0][0]/np.sum(cm[0, :]))
	print("Sensitivity: ", cm[1][1]/np.sum(cm[1, :]))

if __name__ == "__main__":
	main()
