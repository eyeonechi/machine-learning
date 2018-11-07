from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

import numpy as np

inputs = np.array([[i] for i in range(1, 11)])

d = pdist(inputs, 'euclidean')
print(squareform(d))

hc1 = linkage(d, 'single')
dendrogram(hc1, labels=inputs)
plt.show()

hc2 = linkage(d, 'average')
dendrogram(hc2, labels=inputs)
plt.show()

hc3 = linkage(d, 'complete')
dendrogram(hc3, labels=inputs)
plt.show()
