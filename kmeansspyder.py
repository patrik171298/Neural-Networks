import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random
from itertools import chain

def compute_distance(cluster, cluster_center):
    return np.sum(np.power(cluster-cluster_center,2),1)

def random_clustering(data, num_cluster):
    split_points = num_cluster - 1
    split_indices=np.sort(np.random.rand(split_points).tolist())
    split_list = np.round(split_indices*data.shape[0]).astype(int)

    temp = zip(chain([0], split_list), chain(split_list, [None]))
    clusters = list(data[i : j] for i, j in temp)

    #clusters=np.split(data,num_cluster)
    return clusters

def assign_clusters(current_clusters, num_cluster):
    cluster_means=[np.array(cluster.mean(0)) for cluster in current_clusters]
    cluster_mean_distance=np.zeros(num_cluster)
    new_clusters=[[] for i in range(num_cluster)]

    for p, cluster in enumerate(current_clusters):
        cluster_mean_distance=np.zeros((cluster.shape[0],len(current_clusters)))

        for q, individual_cluster_mean in enumerate(cluster_means):
            #Calculating distance between each datapoint in cluster and cluster centers
            cluster_mean_distance[:,q]=compute_distance(current_clusters[p], individual_cluster_mean)
            #Assigning the data point to the cluster center whose distance from the cluster center is minimum of all the cluster centers

        cluster_indices=np.argmin(cluster_mean_distance,1)
        for r, index in enumerate(cluster_indices):
            new_clusters[index].append(cluster[r])

    final_clusters=[np.array(cluster) for cluster in new_clusters]
    return final_clusters

def main():

    dataset=pd.read_excel('data.xlsx').sample(frac=1).reset_index(drop=True)
    X=dataset.iloc[:,:4].values
    X_normalized=(X-X.mean(0))/X[:,0].std(0)

    #X_normalized = X

    num_cluster=int(input("\nPlease enter number of clusters: "))
    cluster_means=np.array(np.zeros(num_cluster))

#Randomly selecting k clusters
    initial_clusters=random_clustering(X_normalized, num_cluster)
    clusters=initial_clusters
    print('Initial clusters')
    for n in range(num_cluster):
        print(clusters[n].shape[0])
    iterations=10

    for i in range(iterations):
        print('\niteration: ',i)
        new_clusters=assign_clusters(clusters, num_cluster)
        for n in range(num_cluster):
            print(clusters[n].shape[0])
        clusters=new_clusters

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors=['r','b','g','y','k','c','m']
    for i, cluster in enumerate(clusters):
        X=cluster[:, 0]
        Y=cluster[:, 1]
        Z=cluster[:, 2]
        ax.scatter(X, Y, Z, c = colors[i], marker = 'o')

    plt.show()


if __name__=="__main__":
    main()
