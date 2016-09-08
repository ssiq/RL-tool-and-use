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

    def reset(self):
        pass


class MonteCarloRobot(Robot):
    def __init__(self, value_approximator=None):
        super(MonteCarloRobot, self).__init__()
        self.value_approximator = value_approximator

    def update(self, observation, reward, done):
        super(MonteCarloRobot, self).update(observation, reward, done)

    def response(self, observation):
        super(MonteCarloRobot, self).response(observation)

    def reset(self):
        pass


class QRobot(Robot):
    def __init__(self, action_space, epsilon=0.5, gamma=0, value_approximator=None,
                 feature=IdentityFeatureExtractor(), verbose=False,
                 epsilon_delta=0.01, replay=0, batch_replay=False):
        super(QRobot, self).__init__()
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.value_approximator = value_approximator
        self.feature = feature
        self.verbose = False
        self.now_features = None
        self.now_action = None
        self.epsilon_delta = epsilon_delta
        self.replay = replay
        self.batch_replay = batch_replay
        self.memory = []

    def reset(self):
        self.now_action = None
        self.now_features = None

    def _best_action(self, features):
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
        return best_action

    def _generate_target_value(self, reward, next_features, done):
        next_action = self._best_action(next_features)
        if not done:
            target_value = reward + self.gamma * self.value_approximator.get_value(next_features, next_action)
        else:
            target_value = reward
        return target_value

    def _update(self, features, action, reward, next_features, done):
        target_value = self._generate_target_value(reward, next_features, done)
        self.value_approximator.update_value(target_value, features, action)

    def _batch_update(self, raw_experience_list):
        experience_list = []
        for features, action, reward, next_features, done in raw_experience_list:
            experience_list.append((self._generate_target_value(reward, next_features, done),
                                    features, action))
        self.value_approximator.batch_update_value(experience_list)

    def update(self, observation, reward, done):
        super(QRobot, self).update(observation, reward, done)
        features = self.feature.transform(observation)
        if self.replay:
            self.memory.append((self.now_features, self.now_action, reward, features, done))
        if not self.batch_replay:
            self._update(self.now_features, self.now_action, reward, features, done)
        if reward > 0:
            self.epsilon = max(self.epsilon-self.epsilon_delta, 0.01)
        if self.replay and len(self.memory) > self.replay:
            if self.batch_replay:
                for i in xrange(self.replay):
                    self._batch_update(random.sample(self.memory, self.replay))
            else:
                for i in xrange(self.replay):
                    features, action, reward, next_features, done = random.choice(self.memory)
                    self._update(features, action, reward, next_features, done)

    def response(self, observation):
        super(QRobot, self).response(observation)
        features = self.feature.transform(observation)
        self.now_features = features

        if flip_coin(self.epsilon):
            self.now_action = random.randint(0, self.action_space-1)
        else:
            self.now_action = self._best_action(features)
        return self.now_action





