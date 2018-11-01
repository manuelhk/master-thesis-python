import matplotlib.pyplot as plt
import numpy as np
import math


def plot_sigmoid():
    x = np.arange(-5., 5., 0.1)
    sig = []
    for item in x:
        sig.append(1 / (1 + math.exp(-item)))

    plt.plot(x, sig)

    plt.ylabel("$\\varphi(x)$")
    plt.xlabel("x")
    plt.grid(True)
    plt.ylim(bottom=0, top=1)
    plt.xlim(left=-5, right=5)

    plt.show()
    pass


def plot_relu():
    x = np.arange(-5., 5., 0.1)
    relu = []
    for item in x:
        relu.append(max(0, item))

    plt.plot(x, relu)

    plt.ylabel("$\\varphi(x)$")
    plt.xlabel("x")
    plt.grid(True)
    plt.ylim(bottom=-1, top=5)
    plt.xlim(left=-5, right=5)

    plt.show()
    pass


plot_sigmoid()
plot_relu()
