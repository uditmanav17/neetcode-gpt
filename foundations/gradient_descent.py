class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        val = init
        grad = lambda x: 2 * x
        for _ in range(iterations):
            val = val - learning_rate * (2 * val)
        return round(val, 5)
            
    