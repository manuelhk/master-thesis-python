import matplotlib.pyplot as plt
import numpy as np
import math


# Methods for plotting figures for the thesis


X = np.arange(-5., 5., 0.1)
Y_LABEL = "$\\varphi(x)$"
X_LABEL = "x"
FONT_SIZE = 16
GRID = True



def plot_sigmoid():
    sig = []
    for item in X:
        sig.append(1 / (1 + math.exp(-item)))

    fig = plt.figure()
    plt.plot(X, sig)

    plt.ylabel(Y_LABEL, fontsize=FONT_SIZE)
    plt.xlabel(X_LABEL, fontsize=FONT_SIZE)
    plt.grid(GRID)

    plt.ylim(bottom=-0.1, top=1.1)
    plt.xlim(left=-5, right=5)

    fig.savefig("output/act_sigmoid.png")
    plt.show()
    pass




def plot_tanh():
    tanh = np.tanh(X)

    fig = plt.figure()
    plt.plot(X, tanh)

    plt.ylabel(Y_LABEL, fontsize=FONT_SIZE)
    plt.xlabel(X_LABEL, fontsize=FONT_SIZE)
    plt.grid(GRID)

    plt.ylim(bottom=-1.2, top=1.2)
    plt.xlim(left=-5, right=5)

    fig.savefig("output/act_tanh.png")
    plt.show()
    pass


def plot_step_fc():
    step = []
    for item in X:
        if item < 0:
            step.append(0)
        else:
            step.append(1)

    fig = plt.figure()
    plt.plot(X, step)

    plt.ylabel(Y_LABEL, fontsize=FONT_SIZE)
    plt.xlabel(X_LABEL, fontsize=FONT_SIZE)
    plt.grid(GRID)

    plt.ylim(bottom=-0.1, top=1.1)
    plt.xlim(left=-5, right=5)

    fig.savefig("output/act_step_fc.png")
    plt.show()
    pass


def plot_relu():
    relu = []
    for item in X:
        relu.append(max(0, item))

    fig = plt.figure()
    plt.plot(X, relu)

    plt.ylabel(Y_LABEL, fontsize=FONT_SIZE)
    plt.xlabel(X_LABEL, fontsize=FONT_SIZE)
    plt.grid(GRID)

    plt.ylim(bottom=-0.5, top=5.5)
    plt.xlim(left=-5, right=5)

    fig.savefig("output/act_relu.png")
    plt.show()
    pass


def plot_softmax():
    softmax = []
    for item in X:
        softmax.append(np.exp(item) / np.exp(X).sum())

    #fig = plt.figure()
    plt.plot(X, softmax)

    plt.ylabel(Y_LABEL, fontsize=FONT_SIZE)
    plt.xlabel(X_LABEL, fontsize=FONT_SIZE)
    plt.grid(GRID)

    plt.ylim(bottom=-0.01, top=0.11)
    plt.xlim(left=-5, right=5)

    #fig.savefig("output/act_relu.png")
    plt.show()
    pass


#plot_sigmoid()
#plot_step_fc()
#plot_tanh()
#plot_relu()
