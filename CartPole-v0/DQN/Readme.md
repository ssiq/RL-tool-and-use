#Description
This file stores the picture from the the method used on the CartPole-v0 problem

##hyperparameters
1. DQN
network architecture
``` python
network = Sequential([
        Dense(input_dim=input_number, output_dim=8),
        Activation('relu'),
        Dense(action_number),
        Activation('linear')
    ])
```
adam, mse
ProritizedMemory(100, 10000)
C=5, batch_size=20, epsilon_delta=0.001
picture 1.jpg

2.CE
```pyhton
robot = GaussLinearCERobot(action_space, input_space,
                               np.zeros(action_space*input_space), np.eye(input_space*action_space),
                               noise_begin=5, noise_delta=0.01, try_number=100, ru=0.2)
```
                                                  
picture 2.jpg
