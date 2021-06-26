import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading
from sklearn.datasets import make_circles
import networkx as nx
import heapq



class KMeans_flex:
    def __init__(self, X, k):
        self.X = X
        self.n = X.shape[0]
        self.k = k
        self.centroids = None
        
    def adj_graph(self,W):
        W_copy = np.copy(W)
        A = []
        for i in range(self.n):
            kth = np.partition(W_copy[i],self.k)[self.k]
            a = []
            for j in range(self.n):
                if W[i][j] <= kth:
                    a.append(1)
                else:
                    a.append(0)
            A.append(np.asarray(a))
        return np.asarray(A)
    def nearest_neighbor(self, X):
        W = []
        for x in self.X:
            distances = [self.dist(c,x) for c in self.X ]
            W.append(distances)        
        W= self.adj_graph(W)
        return W
    def nearest_cluster(self, X):
        centroids_assigned = []
        for x in X:
            distances = [self.dist(c,x) for c in self.centroids ]
            centroids_assigned.append(np.argmin(distances))
        return np.asarray(centroids_assigned)
    
    def dist(self,a,b):
        return np.linalg.norm(a-b, 4)
    def transform(self, X):
        W = self.nearest_neighbor(X)
        G = nx.from_numpy_matrix(W)
        D = np.diag(W.sum(axis=1))
        L = D-W
        e,v = np.linalg.eig(L)
        sorted_e = np.argsort(e.real)
        sorted_v = v[:,sorted_e].real
        sorted_v = sorted_v[:,:self.k]
        return sorted_v
    
    def fit(self):
        np.random.seed(1)
        X_ = self.transform(self.X)
        
        r_id = np.random.choice(np.arange(self.n),size= self.k, replace=False)
        
        self.centroids = X_[r_id, :]
     
        points = self.nearest_cluster(X_)
        while True:
            centroids = []
            for j in range(self.k):
                temp_cent = X_[points==j].mean(axis=0)
                centroids.append(temp_cent)
            self.centroids = np.asarray(centroids)
            new_points = self.nearest_cluster(X_)
            if np.array_equal(new_points, points):
                break
            points = new_points
        return points

n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False) # circle
kmf = KMeans_flex(X , 2)
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
# kmf2 = KMeans_flex(X , 2)
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
# kmf3 = KMeans_flex(X , 2)
# label= kmf3.fit()
# u_labels = np.unique(label)
# for i in u_labels:
#     plt.scatter(kmf3.centroids[i][0], kmf3.centroids[i][1], marker="x", s= 200, c="black")
#     plt.scatter(X[label == i , 0] ,X[label == i , 1] , label = i)
# plt.legend()
# plt.show()