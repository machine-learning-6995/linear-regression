import numpy as np

originW = [1, 10, 20]


# training data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])

num_of_row = X.shape[0]

initial_bias = np.ones((num_of_row, 1))

X = np.hstack([initial_bias, X])

print(X)


def ridge_regression(X, y, lamb):
    # canculate w with lambda = 1.0
    num_of_col = X.shape[1]
    I = np.eye(num_of_col)
    I[0, 0] = 0  # Không áp dụng điều chỉnh lên hệ số tự do (bias term)

    W = np.linalg.inv(X.T @ X + lamb * I) @ X.T @ y

    return W


W = ridge_regression(X, y, 1.0)
Y_pred = X @ W

print(Y_pred)
