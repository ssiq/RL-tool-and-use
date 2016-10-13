import gym
import theano
import theano.tensor as T
import lasagne

from rl_tool.PGRobot import DiscreteMonteCarloPGRobot
from rl_tool.simulator import Simulator


class PolicyNetwork(object):
    def __init__(self, input_number, output_number):
        self.input_number = input_number
        self.output_number = output_number
        input_var = T.fmatrix('x')
        self.network = lasagne.layers.InputLayer((None, input_number), input_var=input_var)
        self.network = lasagne.layers.DenseLayer(self.network, 300)
        self.network = lasagne.layers.DenseLayer(self.network, 300)
        self.network = lasagne.layers.DenseLayer(self.network, 300)
        self.network = lasagne.layers.DenseLayer(self.network, output_number,
                                                 nonlinearity=lasagne.nonlinearities.softmax)
        prediction = lasagne.layers.get_output(network)
        self.predict_function = theano.function([input_var], prediction)
        all_params = lasagne.layers.get_all_params(network, trainable=True)
        param_grads = theano.grad(prediction, all_params)


    def predict(self, state):
        return self.predict_function(state)

    def train_by_grad(self, X, grad):
        pass

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    input_number = 4
    action_number = 2

    network = PolicyNetwork(input_number, action_number)

    robot = DiscreteMonteCarloPGRobot(network, action_number, update_frequency=1)
    env = gym.make('CartPole-v0')
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=1000, max_episode_length=500)