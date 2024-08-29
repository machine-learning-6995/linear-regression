import numpy as np
import pandas
import os
import matplotlib.pyplot as plt


current_path = os.path.abspath(os.getcwd())

df = pandas.read_csv(
    current_path + "/gradient-descent/trainingdata.csv",
    names=["height", "weight"],
    header=0,
)

X = df["height"].to_numpy().reshape(-1, 1)
# print(X)

y = df["weight"].to_numpy().reshape(-1, 1)
# print(y)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)


# X = np.random.rand(1000, 1)
# y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

# one = np.ones((X.shape[0],1))
# Xbar = np.concatenate((one, X), axis = 1)


def cost(w):
    N = Xbar.shape[0]
    return 0.5 / N * np.linalg.norm(y - Xbar.dot(w), 2) ** 2


def gradient(w):
    N = Xbar.shape[0]
    return 1 / N * Xbar.T.dot(Xbar.dot(w) - y)


def myGD(w_init, grad, rate):
    w = [w_init]
    for loop in range(50):
        w_new = w[-1] - rate * grad(w[-1])
        print("=>> loop", loop, "w_new", w_new, "cost", cost(w_new))
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, loop)


w_init = np.array([[2.0], [1.0]])
(w1, loop) = myGD(w_init, gradient, 1)
print("Solution found by GD: w = ", w1[-1].T, ",\nafter %d iterations." % (loop + 1))
