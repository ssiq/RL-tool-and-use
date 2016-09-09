import gym
from rl_tool.simulator import Simulator
from rl_tool.value_approximator import NetworkApproximater
from rl_tool.robot import QRobot
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    input_number = 4
    action_number = 2
    network = Sequential([
        Dense(input_dim=input_number*action_number, output_dim=32),
        Activation('relu'),
        Dropout(0.5),
        Dense(16),
        Activation('relu'),
        Dropout(0.5),
        Dense(1),
        Activation('linear')
    ])
    network.compile(optimizer='rmsprop', loss='mse')
    value_approx = NetworkApproximater(network, action_number)
    robot = QRobot(action_number, gamma=0.9, value_approximator=value_approx,
                   epsilon_delta=0.0025, epsilon=0.9, replay=10, batch_replay=100, max_memory_size=10000)
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=10000)
