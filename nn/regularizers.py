import numpy as np

class L1:
    def __init__(self, lambda_=1e-4):
        self.lambda_ = lambda_

    def __call__(self, w):
        return self.lambda_ * np.abs(w)

class L2:
    def __init__(self, lambda_=1e-4):
        self.lambda_ = lambda_

    def __call__(self, w):
        return self.lambda_ * np.sum(w ** 2)

if __name__ == "__main__":
    w = np.array([[0, 1, 2, 3],
                    [4, 5, 6, 7]])
    reg = L1(0.9)

    print(reg(w))