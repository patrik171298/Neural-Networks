import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(5)
dataset=pd.read_excel('data.xlsx')

learning_rate=0.0001

X=dataset.iloc[:,:2].values
y=dataset.iloc[:,2].values

X_new=(X-X.mean(0))/X[:,0].std(0)
X_normalized=np.hstack((np.array(np.ones(X_new.shape[0]))[:,np.newaxis],X_new))
y_normalized=(y-y.mean())/y.std()

initial_weights=np.random.random(3)
weights=initial_weights

def cost_function(w):
    J=0.5*np.sum(((np.sum(X_normalized*w.T,1)-y_normalized)**2))#/X_normalized.shape[0]
    return J

def gradient(w):
    return np.sum((np.sum(X_normalized*w.T,1)-y_normalized)[:,np.newaxis]*X_normalized,0)#/X_normalized.shape[0]

max_iterations=5000
plt.figure()

iterations=[]
cost_function_value=[]
cost_function_contour=[]
w0,w1,w2=[],[],[]

for i in range(max_iterations):
    weights=weights-learning_rate*gradient(weights)
    print(weights)
    if i%100==0:
        iterations.append(i)
        cost_function_value.append(cost_function(weights))

print(cost_function(weights))


plt.plot(iterations,cost_function_value)

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
plt.show()
