import numpy as np
from collections import deque


class RandomProcess(object):
    def sample(self):
        pass


class NormalMovingAverageProcess(RandomProcess):
    def __init__(self, mu, sigma, M, out_shape):
        self.sample_ = lambda x: np.random.multivariate_normal(mu, sigma, size=x)
        self.prev_M_samples = deque(self.sample_(M))
        self.out_shape = out_shape

    def sample(self):
        r = np.mean(self.prev_M_samples, axis=0)
        r = np.reshape(r, self.out_shape)
        self.prev_M_samples.popleft()
        self.prev_M_samples.append(self.sample_(1))
        return r
