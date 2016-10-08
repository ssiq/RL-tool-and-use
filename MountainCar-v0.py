import gym
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers

from rl_tool.DQN import DQN
from rl_tool.replay_memory import NormalMemory, ProritizedMemory, OneEndMemory

from rl_tool.simulator import Simulator


class TNetwork(Chain):
    def __init__(self, n_in, n_out):
        super(TNetwork, self).__init__(
            L1=L.Linear(n_in, 100),
            L2=L.Linear(100, 100),
            L3=L.Linear(100, 100),
            Q_value=L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32))
        )

    def Q_func(self, x):
        h = F.leaky_relu(self.L1(x))
        h = F.leaky_relu(self.L2(h))
        h = F.leaky_relu(self.L3(h))
        h = self.Q_value(h)
        return h


class Network(object):
    def __init__(self, in_number, out_number):
        self.net = TNetwork(in_number, out_number)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.net)

    def predict(self, X):
        X = np.array(X, np.float32)
        s = Variable(X)
        Q = self.net.Q_func(s)
        return Q.data

    def train_on_batch(self, X, y):
        self.net.zerograds()
        X = np.array(X, np.float32)
        y = np.array(y, np.float32)
        Q = self.net.Q_func(X)
        loss = F.mean_squared_error(Q, Variable(y))
        loss.backward()
        self.optimizer.update()
        return loss.data


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    input_number = 2
    action_number = 3

    network = Network(input_number, action_number)
    replay_memory = NormalMemory(600, 10000)
    robot = DQN(network, action_number, C=2, replay_memory=replay_memory, batch_size=32,
                epsilon_delta=0.001, replay_times=2)
    simulator = Simulator(env, robot, verbose=True, save_path='register/DQN/MountainCar-v0.jpg')
    simulator.run(episode_number=300, max_episode_length=200)