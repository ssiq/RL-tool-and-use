from rl_tool.feature import IdentityFeatureExtractor
from rl_tool.utility import flip_coin
from robot import Robot
import numpy as np
import random
import copy


class DQN(Robot):
    def __init__(self, network, action_number, C, replay_memory,
                 batch_size=32, replay_times=1, gamma=0.99,
                 start_epsilon=1.0, end_epsilon=0.01, epsilon_delta=0.002,
                 feature=IdentityFeatureExtractor()):
        super(DQN, self).__init__()
        self.network = network
        self.target_network = network
        self.action_number = action_number
        self.C = C
        self.replay_memory = replay_memory
        self.feature = feature
        self.old_features = None
        self.old_action = None
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_delta = epsilon_delta
        self.replay_times = replay_times
        self.step_no = 0
        self.loss_list = []

    def reset(self):
        super(DQN, self).reset()

    def _predict(self, features):
        return self.network.predict(features)

    def _predict_target(self, features):
        return self.target_network.predict(features)

    def response(self, observation):
        features = self.feature.transform(observation)
        if flip_coin(self.epsilon):
            action = random.choice(range(0, self.action_number))
        else:
            action = np.argmax(self._predict(np.array([features]))[0])
        self.old_features = features
        self.old_action = action
        return action

    def _train(self, batch):
        train_X = []
        action_list = []
        now_features_list = []
        done_list = []
        reward_list = []
        for old_features, action, now_features, reward, done in batch:
            action_list.append(action)
            now_features_list.append(now_features)
            done_list.append(done)
            reward_list.append(reward)
            train_X.append(old_features)
        train_X = np.array(train_X)
        action_list = np.array(action_list)
        now_features_list = np.array(now_features_list)
        done_list = np.array(done_list)
        reward_list = np.array(reward_list)
        target = self._predict(train_X)
        target[range(self.batch_size), action_list] \
            = reward_list + \
              self.gamma * np.max(self._predict_target(now_features_list), axis=1) * done_list
        train_y = target
        return self.network.train_on_batch(train_X, train_y)

    def update(self, observation, reward, done):
        features = self.feature.transform(observation)
        self.replay_memory.add_sample((self.old_features, self.old_action, features, reward, done), done)
        batch = self.replay_memory.sample(self.batch_size)
        if batch is None:
            return

        loss = []
        for i in xrange(self.replay_times):
            self.step_no += 1
            loss.append(self._train(batch))
            batch = self.replay_memory.sample(self.batch_size)

            if self.C > 1 and self.step_no % self.C == 0:
                self.target_network = copy.deepcopy(self.network)

        self.loss_list.append(np.array(loss).mean())
        if done:
            print 'This Episode\'s mean loss: {}'.format(np.array(self.loss_list).mean())
            self.loss_list = []
            self.epsilon = max(self.epsilon - self.epsilon_delta, self.end_epsilon)
        return np.array(self.loss_list).mean()



