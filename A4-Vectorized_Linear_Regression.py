import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(5)
dataset=pd.read_excel('data.xlsx')
#print(dataset.head)
learning_rate=0.005

X=dataset.iloc[:,:2].values
y=dataset.iloc[:,2].values
#print(y)
#print(X)

X_normalized=(X-X.mean(0))/X[:,0].std(0)
X_new=np.hstack((np.array(np.ones(X_normalized.shape[0]))[:,np.newaxis],X_normalized))
y_normalized=(y-y.mean())/y.std()

def cost_function(w):
    J=0.5*np.sum(((np.sum(X_new*w.T,1)-y_normalized)**2))
    return J

w=np.dot(np.linalg.inv((X_new.T).dot(X_new)),X_new.T).dot(y_normalized)
print("\nFinal weights: \n\n", w)
print("\nCost function value: ", cost_function(w))
