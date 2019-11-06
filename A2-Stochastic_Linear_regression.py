import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(5)
dataset=pd.read_excel('data.xlsx')

learning_rate=0.005

X=dataset.iloc[:,:2].values
y=dataset.iloc[:,2].values

X_normalized=(X-X.mean(0))/X[:,0].std(0)
X_new=np.hstack((np.array(np.ones(X_normalized.shape[0]))[:,np.newaxis],X_normalized))
y_normalized=(y-y.mean())/y.std()

initial_weights=np.array([1,1,1])#np.random.random(3)
#print(initial_weights)
weights=initial_weights

max_iterations=50
def cost_function(w):
    J=0.5*np.sum(((np.sum(X_new*w.T,1)-y_normalized)**2))#/X_new.shape[0]
    return J

def gradient(xi,w,yi):
    return (np.sum(xi.T*w,0)-yi)*xi

k=0
iterations=[]
cost_function_value=[]
cost_function_contour=[]

for i in range(max_iterations):
    for training_example in range(X_new.shape[0]):
        weights=weights-learning_rate*gradient(X_new[training_example], weights, y_normalized[training_example])

        k=k+1
        cost_function_value.append(cost_function(weights))
        iterations.append(k)
    print(cost_function(weights))
print(weights)
plt.figure()
plt.plot(iterations,cost_function_value)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost function')

#Create the Contour
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
#plt.clabel(cp, inline=True, fontsize=10)
plt.title('Contour Plot')
plt.xlabel('w1')
plt.ylabel('w2')
plt.show()
