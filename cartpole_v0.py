import gym
from rl_tool.simulator import Simulator
from rl_tool.value_approximator import DiscreteLinearValueApproximator
from rl_tool.robot import TDZeroRobot

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    value_approx = DiscreteLinearValueApproximator(0.1, (2, 4))
    robot = TDZeroRobot(2, gamma=0.1, value_approximator=value_approx)
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=100)
