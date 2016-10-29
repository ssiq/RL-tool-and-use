import gym
import lasagne
import numpy as np
import theano

from rl_tool.simulator import Simulator
from rl_tool.PGRobot import DDPG
from rl_tool.random_process import NormalMovingAverageProcess
from rl_tool.replay_memory import NormalMemory


class SubDDPG(DDPG):
    def _build_network(self, input_shape, output_shape, state_input_var, action_input_var):
        policy_network = lasagne.layers.InputLayer(shape=(None, np.cumprod(input_shape)[-1]), input_var=state_input_var,
                                                   name='policy')
        policy_network = lasagne.layers.DenseLayer(policy_network, num_units=20)
        policy_network = lasagne.layers.DenseLayer(policy_network, num_units=20)
        policy_network = lasagne.layers.DenseLayer(policy_network, num_units=np.cumprod(output_shape)[-1],)

        Qvalue_network = lasagne.layers.InputLayer(shape=(None, np.cumprod(input_shape)[-1]), input_var=state_input_var,
                                                   name='Qvalue')
        Qvalue_network = lasagne.layers.DenseLayer(Qvalue_network, num_units=20)
        Qvalue_action_input_layer = lasagne.layers.InputLayer(shape=(None, np.cumprod(output_shape)[-1]),
                                                              input_var=action_input_var)
        Qvalue_network = lasagne.layers.ConcatLayer([Qvalue_network, Qvalue_action_input_layer], axis=1)
        Qvalue_network = lasagne.layers.DenseLayer(Qvalue_network, num_units=20)
        Qvalue_network = lasagne.layers.DenseLayer(Qvalue_network, num_units=1)
        theano.printing.pydotprint(lasagne.layers.get_output(policy_network),
                                   outfile="register/DDPG/logreg_pydotprint_policy_prediction.png", var_with_name_simple=True)
        theano.printing.pydotprint(lasagne.layers.get_output(Qvalue_network),
                                   outfile="register/DDPG/logreg_pydotprint_Qvalue_prediction.png",
                                   var_with_name_simple=True)
        return policy_network, Qvalue_network


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    input_number = (2, )
    action_number = (1, )

    process = NormalMovingAverageProcess(np.zeros(action_number), np.eye(np.cumprod(action_number)[0]),
                                         10, action_number)
    robot = SubDDPG(input_number, action_number, process, NormalMemory(100, 100000))
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=10000, max_episode_length=env.spec.timestep_limit)