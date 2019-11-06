import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def cost_function(X,w,y):
    alpha=0.05
    J=0.5*(np.sum(((np.sum(X*w.T,1)-y)**2))+alpha*np.sum(np.abs(w)))#/X_normalized.shape[0]
    return J

def batch_gradient(X,w,y):
    return np.sum((np.sum(X*w.T,1)-y)[:,np.newaxis]*X,0)#/X_normalized.shape[0]

def stochastic_gradient(xi,w,yi):
    return (np.sum(xi.T*w,0)-yi)*xi

def batch_gradient_descent(X,weights_values,y):
    max_iterations=500
    learning_rate=0.00007
    alpha=0.05
    iterations=[]
    cost_function_value=[]

    plt.figure()
    for i in range(max_iterations):
        weights_values=weights_values-0.5*learning_rate*alpha*np.sign(weights_values)-learning_rate*batch_gradient(X,weights_values,y)
        #print(weights_values)
        iterations.append(i)
        cost_function_value.append(cost_function(X,weights_values,y))

        print(cost_function(X,weights_values,y))


    plt.plot(iterations,cost_function_value)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost function')
    plt.show()

def stochastic_gradient_descent(X,weights_values,y):
    k=0
    learning_rate=0.001
    alpha=0.05
    iterations = []
    cost_function_value=[]
    cost_function_contour=[]
    max_iterations=10

    for i in range(max_iterations):
        for training_example in range(X.shape[0]):
            weights_values=weights_values-0.5*learning_rate*alpha*np.sign(weights_values)-learning_rate*stochastic_gradient(X[training_example], weights_values, y[training_example])
            k=k+1
            cost_function_value.append(cost_function(X,weights_values,y))
            iterations.append(k)
        print(cost_function(X,weights_values,y))

    print(weights_values)
    plt.figure()
    plt.plot(iterations,cost_function_value)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost function')
    plt.show()

def main():
    random.seed(5)
    dataset=pd.read_excel('data.xlsx')
    #print(dataset.head)


    X=dataset.iloc[:,:2].values
    y=dataset.iloc[:,2].values
    #print(y)
    #print(X)
    initial_weights=np.random.random(3)
    weights=initial_weights

    X_new=(X-X.mean(0))/X[:,0].std(0)
    X_normalized=np.hstack((np.array(np.ones(X_new.shape[0]))[:,np.newaxis],X_new))
    y_normalized=(y-y.mean())/y.std()

    print('Please input Multiclass Logistic Regression Algorithm:\n'
          '1. Batch Gradient Descent\n'
          '2. Stochastic Gradient Descent\n')
    algo=int(input("Enter Algorithm: "))

    if algo == 1:
        batch_gradient_descent(X_normalized,weights,y_normalized)
    elif algo == 2:
        stochastic_gradient_descent(X_normalized,weights,y_normalized)
    else:
        print("Option not available")

if __name__=="__main__":
    main()
