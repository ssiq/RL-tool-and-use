import gym
import lasagne
import numpy as np
import theano

from rl_tool.simulator import Simulator
from rl_tool.PGRobot import DDPG
from rl_tool.random_process import GuassianRandomProcess, OURandomProcess
from rl_tool.replay_memory import NormalMemory
from rl_tool.DeterminePolicyNetwork import MlpDeterminePolicyNetwork
from rl_tool.ContinousQValueNetwork import MlpContinousQValueNetwork


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    input_number = (2, )
    action_number = (1, )

    # process = GuassianRandomProcess(max_sigma=1, decay_period=800, out_shape=action_number[0])
    process = OURandomProcess(out_shape=action_number[0])
    policy_network = MlpDeterminePolicyNetwork(input_shape=input_number[0],
                                               output_shape=action_number[0],
                                               hidden_sizes=(32, 32),
                                               bn=True)
    q_value_network = MlpContinousQValueNetwork(input_shape=input_number[0],
                                                action_shape=action_number[0],
                                                hidden_sizes=(32, 32),
                                                bn=True,
                                                action_merge_layer=1)
    replay_memory = NormalMemory(100, 100000)
    policy_update_method = lambda loss, params: lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=1e-3)
    q_value_update_method = lambda loss, params: lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=1e-3)
    robot = DDPG(policy_network=policy_network,
                 q_value_network=q_value_network,
                 random_process=process,
                 memory=replay_memory,
                 policy_update_method=policy_update_method,
                 q_value_update_method=q_value_update_method,
                 action_space=env.action_space,
                 batch_size=32,
                 q_value_weight_decay=0.0,
                 policy_weight_decay=0.0)
    simulator = Simulator(env, robot, verbose=True)
    env.monitor.start('/tmp/cartpole-experiment-1')
    simulator.run(episode_number=1200, max_episode_length=env.spec.timestep_limit, render_per_iterations=10)
    env.monitor.close()