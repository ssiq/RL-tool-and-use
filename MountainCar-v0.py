import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, SGD

from rl_tool.DQN import DQN
from rl_tool.replay_memory import NormalMemory, ProritizedMemory

from rl_tool.simulator import Simulator

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    input_number = 2
    action_number = 3

    network = Sequential([
        Dense(input_dim=input_number, output_dim=8),
        Activation('relu'),
        # Dense(16),
        # Activation('relu'),
        # Dense(4),
        # Activation('relu'),
        Dense(action_number),
        Activation('linear')
    ])
    rmsprop = RMSprop(lr=0.001)
    sgd = SGD(clipnorm=1, momentum=0.5)
    network.compile(optimizer='adam', loss='mse')
    replay_memory = NormalMemory(100, 200)
    robot = DQN(network, action_number, C=2, replay_memory=replay_memory, batch_size=20,
                epsilon_delta=0.001, replay_times=10)
    simulator = Simulator(env, robot, verbose=True, save_path='register/DQN/MountainCar-v0.jpg')
    simulator.run(episode_number=1000, max_episode_length=200)