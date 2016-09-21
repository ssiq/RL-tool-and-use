import gym

from rl_tool.CERobot import GaussLinearCERobot
import numpy as np

from rl_tool.simulator import Simulator

if __name__ == '__main__':
    input_space = 4
    action_space = 2
    env = gym.make('CartPole-v1')
    env.monitor.start('/tmp/cartpole-experiment-2')
    robot = GaussLinearCERobot(action_space, input_space,
                               np.zeros(action_space*input_space), np.eye(input_space*action_space),
                               noise_begin=5, noise_delta=0.01, try_number=100, ru=0.2)
    simulator = Simulator(env, robot, verbose=True, save_path='CartPole-v0/DQN/2.jpg')
    simulator.run(episode_number=1000, max_episode_length=500, has_loss=False)
    env.monitor.close()