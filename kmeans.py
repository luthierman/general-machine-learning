import numpy as np
import random as r
import pandas as pd
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, X, k):
        self.X = X
        self.n = X.shape[0]
        self.k = k
        self.centroids = None
        
    def nearest_cluster(self, X):
        centroids_assigned = []
        for x in X:
            distances = [self.dist(c,x) for c in self.centroids ]
            centroids_assigned.append(np.argmin(distances))
        return np.asarray(centroids_assigned)
    def dist(self,a,b):
        return np.linalg.norm(a-b, 2)
    
    def fit(self):
        np.random.seed(1)
        r_id = np.random.choice(np.arange(self.n),size= self.k, replace=False)
        self.centroids = self.X[r_id, :]
        points = self.nearest_cluster(self.X)
        while True:
            centroids = []
            
            for j in range(self.k):
                temp_cent = self.X[points==j].mean(axis=0)

                centroids.append(temp_cent)
            self.centroids = np.asarray(centroids)
            new_points = self.nearest_cluster(self.X)
            if np.array_equal(new_points, points):
                break
            points = new_points
        return points

n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False) # circle
kmf = KMeans(X , 2)
label= kmf.fit()


u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(kmf.centroids[i][0], kmf.centroids[i][1], marker="x", s= 200, c="black")
    plt.scatter(X[label == i , 0] ,X[label == i , 1] , label = i)
plt.legend()
plt.show()

#### UNCOMMENT TO SEE MORE PLOTS AND DIFFERENT DATA ######

# # moons
# from sklearn.datasets import make_moons
# n_samples = 200
# X, y = make_moons(n_samples=n_samples, shuffle=False)
# outer, inner = 0, 1
# kmf2 = KMeans(X , 2)
# label= kmf2.fit()
# u_labels = np.unique(label)
# for i in u_labels:
#     plt.scatter(kmf2.centroids[i][0], kmf2.centroids[i][1], marker="x", s= 200, c="black")
#     plt.scatter(X[label == i , 0] ,X[label == i , 1] , label = i)
# plt.legend()
# plt.show()

# # swirl
# n_samples = 200
# np.random.seed(0)
# t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
# x = t * np.cos(t)
# y = t * np.sin(t)

# X = np.concatenate((x, y))
# X += .7 * np.random.randn(2, n_samples)
# X = X.T
# kmf3 = KMeans(X , 2)
# label= kmf3.fit()
# u_labels = np.unique(label)
# for i in u_labels:
#     plt.scatter(kmf3.centroids[i][0], kmf3.centroids[i][1], marker="x", s= 200, c="black")
#     plt.scatter(X[label == i , 0] ,X[label == i , 1] , label = i)
# plt.legend()
# plt.show()