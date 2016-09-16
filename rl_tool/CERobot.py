from robot import Robot
import numpy as np


class CERobot(Robot):
    '''The abstract base class of the CERobot

    # Argument
        try_number: the number of episode to approximate the score of now distribution
        ru: the ru is the ratio of now sampled parameters should be contained

    # Note
        This class does not implement the three functions which is related to the distribution
        and the approximate function which should be implemented in the subclass
    '''
    def __init__(self, try_number, ru):
        super(CERobot, self).__init__()
        self.try_number = try_number
        self.ru = ru
        self.now_number = 0
        self.now_score = 0
        self.weight_score_list = []
        self.now_weight = self._sample_weight()

    def _sample_weight(self):
        '''
        :return: return a weight sampled from now distribution
        '''
        raise Exception('You should implement the _sample_weight function in the subclass')

    def _get_action(self, observation, weight):
        '''
        :param observation: now observation
        :param weight: the weight used to choose action
        :return: return a action decided by now approximate function
        '''
        raise Exception('You should implement the _get_action function in the subclass')

    def _update_parameter(self, weight_score_list):
        '''
        :param weight_score_list: a list of (weight, score) that should used to update the distribution
        :return: None
        '''
        raise Exception('You should implement the _update_parameter function in the subclass')

    def update(self, observation, reward, done):
        self.now_score += reward
        if done:
            self.now_number += 1
            self.weight_score_list.append((self.now_weight, self.now_number))
            self.now_weight = self._sample_weight()
            self.now_score = 0

        if self.now_number == self.try_number:
            left_number = int(self.try_number * self.ru)
            ws_l = sorted(self.weight_score_list, lambda x, y: cmp(x[1], y[1]))[:-left_number]
            self._update_parameter(ws_l)
            self.weight_score_list = []
            self.now_weight = self._sample_weight()
            self.now_number = 0

    def response(self, observation):
        return self._get_action(observation, self.now_weight)

    def reset(self):
        super(CERobot, self).reset()


class GaussLinearCERobot(CERobot):
    def __init__(self, action_space, input_space, mean, covariance_matrix,
                 noise_begin=0, noise_delta=0, try_number=10, ru=0.9):
        self.action_space = action_space
        self.input_space = input_space
        self.noise = noise_begin
        self.noise_delta = noise_delta
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        super(GaussLinearCERobot, self).__init__(try_number, ru)

    def _get_action(self, observation, weight):
        observation = observation.reshape(-1)
        return np.argmax(weight.dot(observation))

    def _update_parameter(self, weight_score_list):
        self.noise = max(self.noise - self.noise_delta, 0.0)
        weights = np.array([a.reshape(-1) for a, _ in weight_score_list])
        self.mean = weights.mean(axis=0)
        self.covariance_matrix = (weights - self.mean).T.dot(weights-self.mean)/len(weight_score_list) + self.noise

    def _sample_weight(self):
        from numpy.random import multivariate_normal
        return multivariate_normal(self.mean, self.covariance_matrix).reshape(self.action_space, self.input_space)
