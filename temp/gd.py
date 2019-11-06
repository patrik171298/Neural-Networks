import os
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def cost_func(X, Y, B):
    return np.sum((X.dot(B)-Y)**2)/(2*len(Y))

def plot_func(X, Y, w1, w2):
    B = np.array([0, w1, w2])
    return np.sum((X.dot(B)-Y)**2)/(2*len(Y))

def batch_gradient_descent(X, Y, B, alpha, n_iters):
    cost_h = [0]*n_iters
    m = len(Y)

    for it in range(n_iters):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss)/m
        B = B - alpha*gradient
        cost = cost_func(X, Y, B)
        cost_h[it] = cost

    return B, cost_h

def main():
    pwd = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(pwd, 'data.xlsx')
    xl = pd.ExcelFile(f)
    df = xl.parse("Sheet1", header=None, names=['x1', 'x2', 'y'])
    # print(df.head())

    alpha = 0.00000001
    n_iterations = 100

    x1 = df['x1'].values
    x2 = df['x2'].values
    y = df['y'].values

    x0 = np.ones(len(x1))
    X = np.array([x0, x1, x2]).T
    B = np.array([0,0,0])
    Y = np.array(y)
    print(B.shape)
    init_cost = cost_func(X,Y,B)
    print(init_cost)
    B, costs = batch_gradient_descent(X, Y, B, alpha, n_iterations)
    print(B)
    print(costs[-1])

    # fig = plt.figure()
    ax = plt.axes(projection='3d')
    w1 = np.linspace(-6, 6, 100)
    w2 = np.linspace(-6, 6, 100)
    W1, W2 = np.meshgrid(w1, w2)
    J = plot_func(X, Y, W1, W2)

    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('J(w1, w2)')
    ax.plot_surface(W1, W2, J, rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_title('Q1: Linear Regression with Batch Gradient Descent')
    plt.show()

if __name__ == "__main__":
    main()
