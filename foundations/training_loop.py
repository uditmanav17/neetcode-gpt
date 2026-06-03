import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(
        self, 
        X: NDArray[np.float64], 
        y: NDArray[np.float64], 
        epochs: int, lr: float
    ) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n_samples, n_features = X.shape
        
        w = np.zeros(shape=(n_features, 1))
        b = 0
        for _ in range(epochs):
            y_hat = X @ w + b
            # print(f"{y_hat.shape=}")
            diff = y_hat - y.reshape(-1, 1)
            # L = (1/n_samples) * np.sum(np.square(diff))
            dL_dw = 2/n_samples * X.T @ (diff)
            dL_db = 2/n_samples * np.sum(diff)
            # print(f"{dL_dw.shape=}")
            # print(f"{dL_db.shape=}")
            w = w - lr * dL_dw
            b = b - lr * dL_db
        return (np.round(w, 5).squeeze(-1), np.round(b, 5))

