import numpy as np

class L1:
    def __init__(self, lambda_=1e-4):
        self.lambda_ = lambda_

    def __call__(self, w):
        return self.lambda_ * np.sum(np.abs(w))

#objective of regularization: weight decay: push weights to smaller values
class L2:
    def __init__(self, lambda_=1e-4):
        self.lambda_ = lambda_

    def __call__(self, w): #for cost function
        return self.lambda_ * np.sum(np.square(w))
    
    def deriv(self, w, m): #for backpropagation
        return (self.lambda_ / m) * w

class Dropout:
    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
    
    def __call__(self, a):
        m, n = a.shape
        d = np.random.rand(m, n) < self.keep_prob
        a = (a * d) / self.keep_prob #inverted dropout -> result of the cost will still have the same expected value as without drop-out
        
        return a
    
