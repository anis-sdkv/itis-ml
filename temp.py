from Homework1_MyKMeans import MyKMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sem2_iris import show_4_dimention_data

# arrange
flowers = load_iris()
X = flowers['data']

# def generate_blobs():
#     x, y = [], []
#     for i in range(250):
#         x.append(np.random.randint(0, 50))
#         y.append(np.random.randint(0, 50))
#         x.append(100 + np.random.randint(0, 50))
#         y.append(np.random.randint(0, 50))
#         x.append(np.random.randint(0, 50))
#         y.append(100 + np.random.randint(0, 50))
#
#     x = np.array(x)
#     y = np.array(y)
#
#     matrix = [[x[i], y[i]] for i in range(len(x))]
#
#     return np.vstack(matrix)
#
# X = generate_blobs()

# act
kmeans = MyKMeans(3)
kmeans.fit(X, visualize=False)
labels = kmeans.predict(X)

# visualize
# sns.pairplot(X, hue='species')
# plt.scatter(X[:,0], X[:,1], c=labels)
# plt.show()
show_4_dimention_data(X, labels)