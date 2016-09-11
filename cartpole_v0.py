import gym
from rl_tool.simulator import Simulator
from rl_tool.DQN import DQN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, SGD
from rl_tool.replay_memory import NormalMemory, ProritizedMemory

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    input_number = 4
    action_number = 2
    network = Sequential([
        Dense(input_dim=input_number, output_dim=8),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(action_number),
        Activation('linear')
    ])
    rmsprop = RMSprop(lr=0.001)
    sgd = SGD(clipnorm=1, momentum=0.5)
    network.compile(optimizer='adam', loss='mse')
    replay_memory = ProritizedMemory(100, 10000)
    robot = DQN(network, action_number, C=1, replay_memory=replay_memory, batch_size=20, epsilon_delta=0.002)
    simulator = Simulator(env, robot, verbose=True)
    simulator.run(episode_number=10000, max_episode_length=200)
