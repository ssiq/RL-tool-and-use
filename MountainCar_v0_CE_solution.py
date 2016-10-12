import gym

from rl_tool.CERobot import GaussLinearCERobot

from rl_tool.simulator import Simulator

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    input_number = 2
    action_number = 3

    robot = GaussLinearCERobot(action_space=action_number, input_space=input_number,
                               noise_begin=5, noise_delta=0.01, try_number=500, ru=0.2)

    simulator = Simulator(env, robot, verbose=True, save_path='register/DQN/MountainCar-v0.jpg')
    simulator.run(episode_number=1000, max_episode_length=200)
    env.monitor.close()