import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

flowers = load_iris()
X = flowers['data']
target = flowers.target


# print(flowers.target)

def get(dataMatrix, i):
    return list(map(lambda x: x[i], dataMatrix))


def show_4_dimention_data(data, labels):
    fig, axs = plt.subplots(4, 4)
    fig.set_size_inches(200, 200)
    for i in range(4):
        for j in range(4):
            axs[i, j].scatter(data[:, i], data[:, j], c=labels)

    plt.show()
