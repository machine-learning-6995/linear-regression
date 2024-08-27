import numpy as np

# training data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])

# add the initial bias (w0) to each item in X
num_of_row = X.shape[0]
init_bias = np.ones((num_of_row, 1))
X = np.hstack([init_bias, X])

# calculate W vector following the OLS formula
XT = X.T
W = np.linalg.inv(XT @ X) @ XT @ y

# predict
y_pred = X @ W

print(y_pred)
