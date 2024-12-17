import numpy as np
from sklearn.cluster import DBSCAN

from sem5_DBCAN import App


class MyDbscan:
    TYPE_MAIN = 1
    TYPE_BORDER = 0
    TYPE_NOISE = -1
    TYPE_UNCHECKED = -2

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.point_types = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.point_types = np.full(n_samples, MyDbscan.TYPE_UNCHECKED)
        self.labels = np.full(n_samples, -1)

        cluster_id = 0

        for point_index in range(n_samples):
            if self.point_types[point_index] != MyDbscan.TYPE_UNCHECKED:
                continue

            neighbor_indexes = self._find_neighbors(X, point_index)
            if len(neighbor_indexes) < self.min_samples - 1:
                self.point_types[point_index] = MyDbscan.TYPE_NOISE
                self.labels[point_index] = -1
                continue

            self.labels[point_index] = cluster_id
            self.point_types[point_index] = MyDbscan.TYPE_MAIN

            to_check = list(neighbor_indexes)
            for check_index in to_check:
                if self.point_types[check_index] == MyDbscan.TYPE_NOISE:
                    self.point_types[check_index] = MyDbscan.TYPE_BORDER
                    self.labels[check_index] = cluster_id
                if self.point_types[check_index] != MyDbscan.TYPE_UNCHECKED:
                    continue

                self.labels[check_index] = cluster_id
                neighbor_indexes = self._find_neighbors(X, check_index)

                if len(neighbor_indexes) < self.min_samples - 1:
                    self.point_types[check_index] = MyDbscan.TYPE_BORDER
                else:
                    self.point_types[check_index] = MyDbscan.TYPE_MAIN
                    for i in neighbor_indexes:
                        to_check.append(i)

            cluster_id += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def _find_neighbors(self, X, point_index):
        distances = np.linalg.norm(X - X[point_index], axis=1)
        return np.where((distances <= self.eps) & (distances != 0))[0]


if __name__ == "__main__":
    dbscan_eps = 25
    dbscan_min_samples = 5
    app = App([DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples), MyDbscan(dbscan_eps, dbscan_min_samples)])
    app.run()
