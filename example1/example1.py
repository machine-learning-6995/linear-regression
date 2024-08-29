import numpy as np
import pandas
import os
import matplotlib.pyplot as plt


current_path = os.path.abspath(os.getcwd())

print("current_path" + current_path)

df = pandas.read_csv(
    current_path + "/" + "/example1/trainingdata.csv",
    names=["height", "weight"],
    header=0,
)


X = df["height"].to_numpy().reshape(-1, 1)
# print(X)

y = df["weight"].to_numpy().reshape(-1, 1)
# print(y)

# plt.plot(X, y, 'ro')
# plt.axis([140, 190, 45, 75])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

w_0 = w[0][0]
w_1 = w[1][0]
X_pred = np.array([[1, 145], [1, 185]])
y_pred = X_pred @ w
# print(y0)
print(y_pred)

print("Found w = ", w)

# # Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(X_pred, y_pred)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()