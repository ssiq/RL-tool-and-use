import gym
import lasagne

from rl_tool.PGRobot import ContinuousMonteCarloPGRobot
from rl_tool.simulator import Simulator
from rl_tool.PolicyNetwork import ContinuousPolicyNetwork, rmsprop


class PolicyNetwork(ContinuousPolicyNetwork):
    def _build_network(self, input_var):
        self.network = lasagne.layers.InputLayer((None, input_number), input_var=input_var)
        self.network = lasagne.layers.DenseLayer(self.network, 20, nonlinearity=lasagne.nonlinearities.tanh,
                                                 W=lasagne.init.HeNormal())
        self.network = lasagne.layers.DenseLayer(self.network, (self.output_number + self.output_number**2),
                                                 W=lasagne.init.HeNormal(),
                                                 nonlinearity=None)
        return self.network


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    input_number = 2
    action_number = 1

    network = PolicyNetwork(input_number, action_number, rmsprop, learning_rate=0.05)

    robot = ContinuousMonteCarloPGRobot(network, action_number, gamma=1.0, timesteps_per_batch=100000, base_line=False)
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=10000, max_episode_length=env.spec.timestep_limit, done_reward=1)