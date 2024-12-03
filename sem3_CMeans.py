import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from fcmeans import FCM
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

flowers = load_iris()
data = flowers['data']
target = flowers.target

X = data
cmeans = FCM(n_clusters=3)
cmeans.fit(X)
predict = cmeans.predict(X)


# над главной диагональю таргет, под диагональю предикты
def show_4_dimention_data(data, target, labels):
    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            if j > i:
                axs[i, j].scatter(data[:, i], data[:, j], c=target)
            else:
                axs[i, j].scatter(data[:, j], data[:, i], c=labels)

    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.show()

# [if target[i] == labels[i] 1 else 0 for i in range(len(target))]

# show_4_dimention_data(X, target, predict)

accuracy = accuracy_score(target, predict)
print(accuracy)