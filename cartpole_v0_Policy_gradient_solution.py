import gym
import lasagne

from rl_tool.PGRobot import DiscreteMonteCarloPGRobot
from rl_tool.simulator import Simulator
from rl_tool.PolicyNetwork import DiscretePolicyNetwork, rmsprop


class PolicyNetwork(DiscretePolicyNetwork):
    def _build_network(self, input_var):
        self.network = lasagne.layers.InputLayer((None, input_number), input_var=input_var)
        self.network = lasagne.layers.DenseLayer(self.network, 20, nonlinearity=lasagne.nonlinearities.tanh,
                                                 W=lasagne.init.HeNormal())
        self.network = lasagne.layers.DenseLayer(self.network, self.output_number,
                                                 W=lasagne.init.HeNormal(),
                                                 nonlinearity=lasagne.nonlinearities.softmax)
        return self.network


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    input_number = 4
    action_number = 2

    network = PolicyNetwork(input_number, action_number, rmsprop, learning_rate=0.05)

    robot = DiscreteMonteCarloPGRobot(network, action_number, gamma=1.0)
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=5000, max_episode_length=env.spec.timestep_limit, done_reward=1)
