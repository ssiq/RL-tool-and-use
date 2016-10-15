from collections import OrderedDict

import gym
import theano
import theano.tensor as T
import lasagne
import numpy as np

from rl_tool.PGRobot import DiscreteMonteCarloPGRobot
from rl_tool.simulator import Simulator

def rmsprop(loss, params, learning_rate=1.0, rho=0.9, epsilon=1e-9):
    updates = []
    grads = theano.grad(loss, params)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates.append((accu, accu_new))
        updates.append((param, param + (learning_rate * grad /
                                T.sqrt(accu_new + epsilon))))

    return updates

class PolicyNetwork(object):
    def __init__(self, input_number, output_number):
        from theano import pp
        self.input_number = input_number
        self.output_number = output_number
        input_var = T.fmatrix('x')
        self.network = lasagne.layers.InputLayer((None, input_number), input_var=input_var)
        self.network = lasagne.layers.DenseLayer(self.network, 20, nonlinearity=lasagne.nonlinearities.tanh,
                                                 W=lasagne.init.HeNormal())
        self.network = lasagne.layers.DenseLayer(self.network, output_number,
                                                 W=lasagne.init.HeNormal(),
                                                 nonlinearity=lasagne.nonlinearities.softmax)
        prediction = lasagne.layers.get_output(self.network)
        print pp(prediction)
        self.predict_function = theano.function([input_var], prediction, allow_input_downcast=True)
        all_params = lasagne.layers.get_all_params(self.network, trainable=True)
        action_list = T.ivector('action_list')
        reward_list = T.fvector('reward_list')
        N = input_var.shape[0]
        loss_function = T.log(prediction[T.arange(N), action_list]).dot(reward_list)/N
        updates = rmsprop(loss=loss_function, params=all_params,
                          learning_rate=0.05)
        self.update_function = theano.function([input_var, action_list, reward_list],
                                               loss_function, updates=updates, allow_input_downcast=True)

    def predict(self, state):
        return self.predict_function(state)

    def train_by_grad(self, X, action, reward):
        return self.update_function(X, action, reward)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    input_number = 4
    action_number = 2

    network = PolicyNetwork(input_number, action_number)

    robot = DiscreteMonteCarloPGRobot(network, action_number, gamma=1.0)
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=100000, max_episode_length=env.spec.timestep_limit, done_reward=1)
