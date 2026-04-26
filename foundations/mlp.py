import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(
        self, 
        x: NDArray[np.float64], 
        weights: List[NDArray[np.float64]], 
        biases: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        N = len(weights)
        new_x = x
        for idx in range(N):
            new_x = new_x @ weights[idx] + biases[idx]
            if idx != (N-1):
                new_x =np.maximum(0, new_x)
        return np.round(new_x, 5)

