import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from cource_ml.Homework1_MyKMeans import MyKMeans

x, y = [], []
for i in range(250):
    x.append(np.random.randint(0, 50))
    y.append(np.random.randint(0, 50))
    x.append(100 + np.random.randint(0, 50))
    y.append(np.random.randint(0, 50))
    x.append(np.random.randint(0, 50))
    y.append(100 + np.random.randint(0, 50))

# plt.scatter(x, y)
# plt.show()

x = np.array(x)
y = np.array(y)

matrix = [[x[i], y[i]] for i in range(len(x))]
# ===
# kmeans = KMeans(n_clusters=3)
#
# kmeans.fit(matrix)
# labels = kmeans.labels_
# ===
kmeans = MyKMeans(n_clusters=3)
kmeans.fit(matrix)
labels = kmeans.labels
# ===
plt.scatter(x, y, c=labels)
plt.show()

# print(kmeans.predict([[50, 50], [150, 150]]))
print(labels)
