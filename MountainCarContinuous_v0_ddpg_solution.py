import gym
import lasagne
import numpy as np
import theano

from rl_tool.simulator import Simulator
from rl_tool.PGRobot import DDPG
from rl_tool.random_process import NormalMovingAverageProcess
from rl_tool.replay_memory import NormalMemory
from rl_tool.DeterminePolicyNetwork import MlpDeterminePolicyNetwork
from rl_tool.ContinousQValueNetwork import MlpContinousQValueNetwork


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    input_number = (2, )
    action_number = (1, )

    process = NormalMovingAverageProcess(np.zeros(action_number), np.eye(np.cumprod(action_number)[0]),
                                         10, action_number)
    policy_network = MlpDeterminePolicyNetwork(input_shape=input_number[0],
                                               output_shape=action_number[0],
                                               hidden_sizes=(32, 32, 32))
    q_value_network = MlpContinousQValueNetwork(input_shape=input_number[0],
                                                action_shape=action_number[0],
                                                hidden_sizes=(32, 32, 32))
    replay_memory = NormalMemory(100, 100000)
    policy_update_method = lambda loss, params: lasagne.updates.adam(loss_or_grads=loss, params=params)
    q_value_update_method = lambda loss, params: lasagne.updates.adam(loss_or_grads=loss, params=params)
    robot = DDPG(policy_network=policy_network,
                 q_value_network=q_value_network,
                 random_process=process,
                 memory=replay_memory,
                 policy_update_method=policy_update_method,
                 q_value_update_method=q_value_update_method,
                 )
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=1200, max_episode_length=env.spec.timestep_limit, render_per_iterations=10)