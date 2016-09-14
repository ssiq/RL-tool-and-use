#Description
This file stores the picture from the DQN do on the CartPole-v0 problem

##hyperparameters
1. 
network architecture
``` python
network = Sequential([
        Dense(input_dim=input_number, output_dim=8),
        Activation('relu'),
        Dropout(0.5),
        Dense(16),
        Activation('relu'),
        Dropout(0.5),
        Dense(action_number),
        Activation('linear')
    ])
```
adam, mse
ProritizedMemory(100, 10000)
C=5, batch_size=20, epsilon_delta=0.001
picture 