import numpy as np
from typing import Tuple, List


class Solution:
    def __init__(self):
        self.running_mean = None
        self.running_var = None
        pass

    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        running_mean = np.asarray(running_mean)
        running_var = np.asarray(running_var)
        x = np.asarray(x)
        gamma = np.asarray(gamma)
        beta = np.asarray(beta)

        if training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        y = gamma * x_hat + beta
        return (
            np.round(y, 4).tolist(), 
            np.round(running_mean, 4).tolist(), 
            np.round(running_var, 4).tolist()
        )
            

