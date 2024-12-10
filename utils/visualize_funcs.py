import matplotlib.pyplot as plt


def visualize_n_dimension_data(X, Y):
    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            axs[i, j].scatter(X[:, i], X[:, j], c=Y)

    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.show()


