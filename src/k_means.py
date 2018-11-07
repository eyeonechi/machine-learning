# adapted from Data Science from Scratch by Joel Grus, 2015

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import random

'''k-means clustering'''
class KMeans:

    def __init__(self, k, means=None, display=False):
        self.k = k         # number of clusters
        self.means = means # means of clusters
        self.display = display

    '''returns the index of the cluster closest to the input'''
    def classify(self, record):
        return min(range(self.k), key=lambda i: np.linalg.norm(record - self.means[i]))

    def train(self, inputs):
        if self.means == None:
            self.means = inputs[np.random.choice(np.shape(inputs)[0], self.k)]
        self.assignments = None
        if self.display:
            step = 0
            print('step', step)
            print('data points:', inputs)
            print('assignments:', self.assignments)
            print('centroids:', self.means, '\n')
        while True:
            # find new assignments
            new_assignments = list(map(self.classify, inputs))
            # if no assignments have changed, we're done
            if self.assignments == new_assignments:
                if self.display:
                    step += 1
                    print('step', step)
                    print('data points:', inputs)
                    print('assignments:', self.assignments)
                    print('centroids:', self.means, '\n')
                return
            # otherwise, keep the new assignments
            self.assignments = new_assignments
            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, self.assignments) if a == i]
                # avoid division by zero if i_points is empty
                if i_points:
                    self.means[i] = np.mean(i_points, axis=0)
            if self.display:
                step += 1
                print('step', step)
                print('data points:', inputs)
                print('assignments:', self.assignments)
                print('centroids:', self.means, '\n')

''' finds the total squared error from k-means clustering the inputs '''
def squared_clustering_errors(inputs, k, clusterer=None):
    if clusterer == None:
        clusterer = KMeans(k)
        clusterer.train(inputs)
    sse = sum(np.linalg.norm(input - clusterer.means[cluster]) for input, cluster in zip(inputs, clusterer.assignments))
    print('sse:', sse, 'with k:', k)
    return sse

def plot_squared_clustering_errors(inputs, k1, k2):
    ks = range(k1, k2 + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]
    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel('k')
    plt.ylabel('total squared error')
    plt.show()

def main():
    random.seed(0)

    print('distance=Euclidean, k=2, random_seeds={1, 2}')
    inputs = np.array(range(1, 11))
    km1 = KMeans(2, [1, 2], display=True)
    km1.train(inputs)
    squared_clustering_errors(inputs, 2, km1)

    print('distance=Euclidean, k=2, random_seeds={2, 9}')
    inputs = np.array(range(1, 11))
    km2 = KMeans(2, [2, 9], display=True)
    km2.train(inputs)
    squared_clustering_errors(inputs, 2, km2)

    print('sse calculation')
    inputs = np.random.choice(100, 20)
    plot_squared_clustering_errors(inputs, 2, 20)

    print('dataset example 1')
    input2d = np.genfromtxt('../data/2d-dataset.csv', delimiter=',')
    # cluster and plot assignments and centroids
    clusterer = KMeans(3)
    clusterer.train(input2d)
    centroids = np.array(clusterer.means)
    assignments = list(map(clusterer.classify, input2d))
    # generate string color list from cluster assignments
    colors = ['red', 'green', 'blue']
    c_labels = [colors[grp] for grp in assignments]
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(input2d[:,0], input2d[:,1], c=c_labels)
    ax.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, color='yellow')
    plt.show()

    print('dataset example 2')
    input3d = np.genfromtxt('../data/3d-dataset.csv', delimiter=',')
    # cluster and plot assignments and centroids
    clusterer = KMeans(3)
    clusterer.train(input3d)
    centroids = np.array(clusterer.means)
    assignments = list(map(clusterer.classify, input3d))
    # generate string color list from cluster assignments
    c_labels = [colors[grp] for grp in assignments]
    fig = plt.figure(2)
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.scatter(input3d[:,0], input3d[:,1], input3d[:,2], c=c_labels, marker='o')
    ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], marker='*', s=200, color='yellow')
    plt.show()

if __name__ == '__main__':
    main()
