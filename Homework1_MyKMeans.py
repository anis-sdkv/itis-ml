import numpy as np
import matplotlib.pyplot as plt


# Домашка до 3 декабря
class MyKMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=42, tol=0.1):
        self.n_clusters = n_clusters
        self.centroids = None
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, init_method="k-means++", visualize=False):
        if X is None or X.size == 0:
            raise ValueError("Input data is empty")

        if init_method == "k-means++":
            self._init_centroids_kmeans_pp(X)
        elif init_method == "random":
            self._init_centroids_random(X)
        else:
            raise ValueError("unknown init method")

        for iteration in range(self.max_iter):
            diffs = X[:, np.newaxis] - self.centroids
            # broadcasting (n_samples, 1, n_features) - (n_clusters, n_features) = (n_samples, n_clusters, n_features)
            distances = np.linalg.norm(diffs, axis=2)  # (n_samples, n_clusters)
            labels = np.argmin(distances, axis=1)  # (n_samples)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) <= self.tol):
                break
            self.centroids = new_centroids
            if visualize:
                self._visualize_centroids(X, labels, iteration, 3000)

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")

        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def get_optimal_cluster_num(self, X):

        return 1

    def inertia(self):
        pass

    def _init_centroids_kmeans_pp(self, X):
        # первый элемент - случайный из датасета
        centroids = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            # Для каждой точки вычисляем минимальное расстояние до уже выбранных центроидов
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
            # Выбираем следующий центроид с вероятностью, пропорциональной квадрату расстояния
            probs = distances ** 2
            probs /= probs.sum()
            next_centroid = X[np.random.choice(X.shape[0], p=probs)]
            centroids.append(next_centroid)

        self.centroids = np.array(centroids)

    def _init_centroids_random(self, X):
        np.random.seed(self.random_state)
        self.centroids = np.random.randint(
            np.min(X),
            np.max(X),
            size=(self.n_clusters, len(X[0]))
        )

    def _visualize_centroids(self, X, labels, iteration, pause_ms):
        plt.figure(figsize=(8, 6))

        # Визуализация точек, окрашенных в зависимости от кластера
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='k')

        # Визуализация центроидов
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

        # Подписи
        plt.title(f'Iteration {iteration + 1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        plt.draw()  # Обновление графика
        plt.pause(pause_ms / 1000)  # Задержка

        plt.clf()  # Очистка графика для следующего шага
        plt.close()
