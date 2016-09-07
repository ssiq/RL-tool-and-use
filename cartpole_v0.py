import gym
from rl_tool.simulator import Simulator
from rl_tool.value_approximator import DiscreteLinearValueApproximator
from rl_tool.robot import QRobot

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    value_approx = DiscreteLinearValueApproximator(0.00025, (2, 4))
    robot = QRobot(2, gamma=0.9, value_approximator=value_approx, epsilon_delta=0.0001, epsilon=0.9, replay=100)
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=10000)
