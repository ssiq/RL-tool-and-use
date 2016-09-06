from feature import IdentityFeatureExtractor
from utility import flip_coin
import random


class Robot(object):
    def __init__(self):
        pass

    def response(self, observation):
        pass

    def update(self, observation, reward, done):
        pass


class MonteCarloRobot(Robot):
    def __init__(self, value_approximator=None):
        super(MonteCarloRobot, self).__init__()
        self.value_approximator = value_approximator

    def update(self, observation, reward, done):
        super(MonteCarloRobot, self).update(observation, reward, done)

    def response(self, observation):
        super(MonteCarloRobot, self).response(observation)


class TDZeroRobot(Robot):
    def __init__(self, action_space, epsilon=0.5, gamma=0, lamda=0, value_approximator=None, feature=IdentityFeatureExtractor()):
        super(TDZeroRobot, self).__init__()
        self.action_space = action_space
        self.epsilon = epsilon
        self.lamda = lamda
        self.gamma = gamma
        self.value_approximator = value_approximator
        self.feature = feature
        self.now_features = None
        self.now_action = None

    def update(self, observation, reward, done):
        super(TDZeroRobot, self).update(observation, reward, done)
        features = self.feature.transform(observation)
        next_action = self.response(features)
        target_value = reward + self.value_approximator.get_value(features, next_action)
        self.value_approximator.update_value(target_value, self.now_features, self.now_action)

    def response(self, observation):
        super(TDZeroRobot, self).response(observation)
        features = self.feature.transform(observation)
        self.now_features = features

        if flip_coin(self.epsilon):
            self.now_action = random.randint(0, self.action_space-1)
        else:
            best_value = None
            best_action = None
            for i in xrange(self.action_space):
                if best_value is None:
                    best_value = self.value_approximator.get_value(features, i)
                    best_action = i
                else:
                    value = self.value_approximator.get_value(features, i)
                    if value > best_value:
                        best_value = value
                        best_action = i
            self.now_action = best_action
        return self.now_action





