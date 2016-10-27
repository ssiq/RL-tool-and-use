import gym
import lasagne
import numpy as np

from rl_tool.simulator import Simulator
from rl_tool.PGRobot import DDPG
from rl_tool.random_process import NormalMovingAverageProcess
from rl_tool.replay_memory import NormalMemory


class SubDDPG(DDPG):
    def _build_network(self, input_shape, output_shape, state_input_var, action_input_var):
        pass


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    input_number = 2
    action_number = 1

    process = NormalMovingAverageProcess(np.zeros(action_number), np.eye(action_number), 10)
    robot = SubDDPG(input_number, action_number, process, NormalMemory(100, 100000))
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=10000, max_episode_length=env.spec.timestep_limit, done_reward=1)