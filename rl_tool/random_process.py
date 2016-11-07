import numpy as np
from collections import deque


class RandomProcess(object):
    '''
    The class is created to explore in the determine policy gradient
    '''
    def sample(self, t):
        pass


class GuassianRandomProcess(RandomProcess):
    def __init__(self, min_sigma=0.1, max_sigma=1, decay_period=10000, out_shape=4):
        self.min_sigma, self.max_sigma, self.decay_period, self.out_shape = \
            min_sigma, max_sigma, decay_period, out_shape

    def sample(self, t):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1, t)/self.decay_period
        return np.random.normal(size=self.out_shape) * sigma


class OURandomProcess(RandomProcess):
    def __init__(self, mu=0, theta=0.15, sigma=0.3, out_shape=4):
        self.mu, self.theta, self.sigma, self.out_shape = \
            mu, theta, sigma, out_shape
        self.state = np.ones(self.out_shape) * self.mu

    def sample(self, t):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
