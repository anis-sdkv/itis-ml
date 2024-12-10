import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from PIL import Image

from cource_ml.sem3_CMeans import predict


# Домашка до 3 декабря
class MyKMeans:
    class InitMethods:
        RANDOM = "random"
        MAX_DISTANCE = "max-distance"
        KMEANS_PP = "k-means++"

    class VisualizeModes:
        GIF = "gif"
        PLOT = "plot"

    OUT_DIR = "out"

    def __init__(self, n_clusters, max_iter=100, random_state=42, tol=0.1):
        if n_clusters == 0:
            raise ValueError("n_clusters should be greather than 0")
        self.n_clusters = n_clusters
        self.centroids = None
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, init_method=InitMethods.MAX_DISTANCE, visualize=False, visualize_mode=VisualizeModes.PLOT):
        if X is None or X.size == 0:
            raise ValueError("Input data is empty")

        if init_method == MyKMeans.InitMethods.RANDOM:
            self._init_centroids_random(X)
        elif init_method == MyKMeans.InitMethods.MAX_DISTANCE:
            self._init_centroids_max_distance(X)
        elif init_method == MyKMeans.InitMethods.KMEANS_PP:
            self._init_centroids_kmeans_pp(X)
        else:
            raise ValueError("unknown init method")

        labels = None
        for iteration in range(self.max_iter):
            diffs = X[:, np.newaxis] - self.centroids
            # broadcasting (n_samples, 1, n_features) - (n_clusters, n_features) = (n_samples, n_clusters, n_features)
            distances = np.linalg.norm(diffs, axis=2)  # (n_samples, n_clusters)
            labels = np.argmin(distances, axis=1)  # (n_samples)
            if visualize:
                self._visualize_centroids(X, labels, iteration, 3000, visualize_mode)

            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) <= self.tol):
                break
            self.centroids = new_centroids

        if visualize:
            self._visualize_centroids(X, labels, -1, 3000, visualize_mode)

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")

        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def calculate_inertia(self, X):
        """
        Рассчитывает инерцию (сумма квадратов расстояний внутри кластеров)
        """
        if self.centroids is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'inertia'.")

        inertia = 0
        Y = self.predict(X)
        for i in range(len(X)):
            label = Y[i]
            distance = np.linalg.norm(X[i] - self.centroids[label])
            inertia += distance ** 2
        return inertia

    @staticmethod
    def get_optimal_cluster_num_elbow(X, n_max_clusters=10, visualize=False):
        inertia_list = []
        for i in range(1, n_max_clusters):
            kmeans = MyKMeans(n_clusters=i)
            kmeans.fit(X)
            inertia_list.append(kmeans.calculate_inertia(X))

        if visualize:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, n_max_clusters), inertia_list, marker='o')
            plt.title('Метод локтя для выбора оптимального количества кластеров')
            plt.xlabel('Количество кластеров')
            plt.ylabel('Инерция')
            plt.xticks(range(1, 11))
            plt.show()

        # вычисление локтя по формуле из книги
        diffs = [abs(inertia_list[i] - inertia_list[i + 1]) for i in range(0, len(inertia_list) - 1)]
        rel = [diffs[i + 1] / diffs[i] for i in range(0, len(diffs) - 1)]
        optimal_i = np.argmin(rel) + 2
        return optimal_i

    def _init_centroids_max_distance(self, X):
        diffs = X[:, np.newaxis] - X
        pairwise_distances = np.linalg.norm(diffs, axis=2)
        max_distance_points_indexes = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
        selected_centroids = [X[i] for i in max_distance_points_indexes]
        for i in range(2, self.n_clusters):
            diffs = X[:, np.newaxis] - np.array(selected_centroids)
            distances = np.linalg.norm(diffs, axis=2)

            # Находим минимальное расстояние до ближайшего центра для каждой точки
            min_distances = np.min(distances, axis=1)   
    
            # Выбираем точку, у которой это минимальное расстояние максимально  
            next_centroid_index = np.argmax(min_distances)  
            selected_centroids.append(X[next_centroid_index])   
    
        self.centroids = np.array(selected_centroids)   

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

    def _visualize_centroids(self, X, Y, iteration, pause_ms, mode):
        n_features = np.shape(X)[1]
        fig, axs = plt.subplots(n_features, n_features)
        fig.set_figwidth(10)
        fig.set_figheight(10)

        if iteration == -1:
            plt.suptitle(f'Final result')
        elif iteration == 0:
            plt.suptitle(f'Initial centroids')
        else:
            plt.suptitle(f'Iteration {iteration}')

        for i in range(n_features):
            for j in range(n_features):
                axs[i, j].scatter(X[:, i], X[:, j], c=Y)
                axs[i, j].scatter(
                    self.centroids[:, i],
                    self.centroids[:, j],
                    c='red',
                    marker='X',
                    s=100,
                    label="Centroids"
                )
        if mode == MyKMeans.VisualizeModes.GIF:
            if iteration == 0:
                if os.path.exists(MyKMeans.OUT_DIR):
                    shutil.rmtree(MyKMeans.OUT_DIR)
                os.mkdir(MyKMeans.OUT_DIR)
            plt.savefig(f'out/temp_plot_{int(time.time() * 1000)}.png', format='png', dpi=300, bbox_inches='tight',
                        transparent=True)
            if iteration == -1:
                self._generate_gif(pause_ms)
        elif mode == MyKMeans.VisualizeModes.PLOT:
            if iteration == -1:
                plt.show()
            else:
                plt.draw()  # Обновление графика
                plt.pause(pause_ms / 1000)  # Задержка

                plt.clf()  # Очистка графика для следующего шага
                plt.close()
        else:
            raise Exception("unknown mode")

    def _generate_gif(self, frame_duration):
        image_files = [f for f in os.listdir(MyKMeans.OUT_DIR) if f.endswith('.png')]
        image_files.sort()
        images = [Image.open(os.path.join(MyKMeans.OUT_DIR, file)) for file in image_files]
        images_with_white_bg = []
        for img in images:
            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            white_bg.paste(img, (0, 0), img)
            images_with_white_bg.append(white_bg)
        images = images_with_white_bg
        images[0].save(
            f'{MyKMeans.OUT_DIR}/output.gif',
            save_all=True,
            append_images=images[1:],
            duration=frame_duration,
            loop=0
        )

        for file in image_files:
            file_path = os.path.join(MyKMeans.OUT_DIR, file)
            os.remove(file_path)


# arrange
def generate_blobs():
    x, y = [], []
    for i in range(250):
        x.append(np.random.randint(0, 50))
        y.append(np.random.randint(0, 50))
        x.append(100 + np.random.randint(0, 50))
        y.append(np.random.randint(0, 50))
        x.append(np.random.randint(0, 50))
        y.append(100 + np.random.randint(0, 50))

    x = np.array(x)
    y = np.array(y)

    matrix = [[x[i], y[i]] for i in range(len(x))]

    return np.vstack(matrix)

flowers = load_iris()
X = flowers['data']

# act
optimal_clusters_count = MyKMeans.get_optimal_cluster_num_elbow(X, visualize=True)
kmeans = MyKMeans(optimal_clusters_count)
kmeans.fit(X, visualize=True, visualize_mode=MyKMeans.VisualizeModes.PLOT)
Y = kmeans.predict(X)
