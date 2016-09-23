#CartPole-v0
This part stores the picture from the the method used on the CartPole-v0 problem

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


