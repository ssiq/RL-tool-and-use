import gym
from rl_tool.simulator import Simulator
from rl_tool.DQN import DQN
from keras.models import Sequential
from keras.layers import Dense, Activation
from rl_tool.replay_memory import NormalMemory

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    input_number = 4
    action_number = 2
    network = Sequential([
        Dense(input_dim=input_number, output_dim=16),
        Activation('relu'),
        Dense(8),
        Activation('relu'),
        Dense(action_number),
        Activation('linear')
    ])
    network.compile(optimizer='adam', loss='mse')
    replay_memory = NormalMemory(100, 10000)
    robot = DQN(network, action_number, C=5, replay_memory=replay_memory, batch_size=50)
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=10000)
