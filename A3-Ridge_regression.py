import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(5)
dataset=pd.read_excel('data.xlsx')
#print(dataset.head)


X=dataset.iloc[:,:2].values
y=dataset.iloc[:,2].values
#print(y)
#print(X)
initial_weights=np.random.random(3)
weights=initial_weights

X_normalized=(X-X.mean(0))/X[:,0].std(0)
X_new=np.hstack((np.array(np.ones(X_normalized.shape[0]))[:,np.newaxis],X_normalized))
y_normalized=(y-y.mean())/y.std()

def cost_function(w):
    alpha=0.05
    J=0.5*(np.sum(((np.sum(X_new*w.T,1)-y_normalized)**2))+alpha*np.sum(w**2))#/X_new.shape[0]
    return J

def batch_gradient(w):
    return np.sum((np.sum(X_new*w.T,1)-y_normalized)[:,np.newaxis]*X_new,0)#/X_new.shape[0]

def stochastic_gradient(xi,w,yi):
    return (np.sum(xi.T*w,0)-yi)*xi

iterations=[]
cost_function_value=[]
cost_function_contour=[]

def batch_gradient_descent(weights_values):
    max_iterations=2000
    learning_rate=0.0001
    alpha=0.05
    plt.figure()
    for i in range(max_iterations):
        weights_values=(1-learning_rate*alpha)*weights_values-learning_rate*batch_gradient(weights_values)
        print(weights)

        iterations.append(i)
        cost_function_value.append(cost_function(weights_values))

    print(cost_function(weights_values))


    plt.plot(iterations,cost_function_value)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost function')

def stochastic_gradient_descent(weights_values):
    k=0
    learning_rate=0.001
    alpha=0.05
    iterations=[]
    cost_function_value=[]
    cost_function_contour=[]
    max_iterations=125

    for i in range(max_iterations):
        for training_example in range(X_new.shape[0]):
            weights_values=(1-learning_rate*alpha)*weights_values-learning_rate*stochastic_gradient(X_new[training_example], weights_values, y_normalized[training_example])
            k=k+1
            cost_function_value.append(cost_function(weights_values))
            iterations.append(k)
        print(cost_function(weights_values))
    print(weights_values)
    plt.figure()
    plt.plot(iterations,cost_function_value)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost function')

def main():

    print('Please input Multiclass Logistic Regression Algorithm:\n'
          '1. Batch Gradient Descent\n'
          '2. Stochastic Gradient Descent\n')
    algo=int(input("Enter Algorithm: "))

    if algo == 1:
        batch_gradient_descent(weights)
    elif algo == 2:
        stochastic_gradient_descent(weights)
    else:
        print("Option not available")

    plt.figure()

    w11=np.linspace(-1,1,100)
    w22=np.linspace(-1,1,100)
    w1_mesh,w2_mesh=np.meshgrid(w11,w22)

    bias=np.array(np.ones((100,100)))
    grid=np.dstack([bias,w1_mesh,w2_mesh])
    new_grid=grid.reshape(-1,3)
    l=np.array([cost_function(b) for b in new_grid])
    lx=l.reshape(100,100)

    plt.contour(w1_mesh, w2_mesh, lx, colors='black')
    plt.title('Contour Plot')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.show()

if __name__=="__main__":
    main()
