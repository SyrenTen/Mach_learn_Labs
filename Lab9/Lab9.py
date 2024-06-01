import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import time
from sklearn.decomposition import PCA

digits = load_digits()
data = digits.data
true_label = digits.target

data_scaled = scale(data)
n_clusters = len(np.unique(true_label))


class CustomKMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        self.centr = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            self.labels_ = self.assign_clusters(X)
            new_centroids = self.compute_centr(X)
            if np.all(np.abs(new_centroids - self.centr) < self.tol):
                break
            self.centr = new_centroids
        return self

    def assign_clusters(self, X):
        dist = np.linalg.norm(X[:, np.newaxis] - self.centr, axis=2)
        return np.argmin(dist, axis=1)

    def compute_centr(self, X):
        return np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])


start_time = time.time()
cust_kmeans = CustomKMeans(n_clusters=n_clusters)
cust_kmeans.fit(data_scaled)
custom_time = time.time() - start_time
c_ari = adjusted_rand_score(true_label, cust_kmeans.labels_)
c_ami = adjusted_mutual_info_score(true_label, cust_kmeans.labels_)

start_time = time.time()
kmeans_pp = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=0)
kmeans_pp.fit(data_scaled)
pp_time = time.time() - start_time
pp_ari = adjusted_rand_score(true_label, kmeans_pp.labels_)
pp_ami = adjusted_mutual_info_score(true_label, kmeans_pp.labels_)

start_time = time.time()
kmeans_rand = KMeans(n_clusters=n_clusters, init='random', n_init=10, random_state=0)
kmeans_rand.fit(data_scaled)
rand_time = time.time() - start_time
rand_ari = adjusted_rand_score(true_label, kmeans_rand.labels_)
rand_ami = adjusted_mutual_info_score(true_label, kmeans_rand.labels_)

print(f'KMeans: ARI = {c_ari:.4f}, AMI = {c_ami:.4f}, Time = {custom_time:.4f}')
print(f'KMeans++: ARI = {pp_ari:.4f}, AMI = {pp_ami:.4f}, Time = {pp_time:.4f}')
print(f'Random KMeans: ARI = {rand_ari:.4f}, AMI = {rand_ami:.4f}, Time = {rand_time:.4f}')


def plot_clusters(data, labels, centroids, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
    plt.title(title)
    plt.show()

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
cust_centr_pca = pca.transform(cust_kmeans.centr)
kmeans_pp_pca = pca.transform(kmeans_pp.cluster_centers_)
kmeans_rand_pca = pca.transform(kmeans_rand.cluster_centers_)

plot_clusters(data_pca, cust_kmeans.labels_, cust_centr_pca, 'KMeans')
plot_clusters(data_pca, kmeans_pp.labels_, kmeans_pp_pca, 'KMeans++')
plot_clusters(data_pca, kmeans_rand.labels_, kmeans_rand_pca, 'Random KMeans')
