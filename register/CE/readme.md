CE
1.solve the catpolev1 problem
```pyhton
robot = GaussLinearCERobot(action_space, input_space,
                               np.zeros(action_space*input_space), np.eye(input_space*action_space),
                               noise_begin=5, noise_delta=0.01, try_number=100, ru=0.2)
```
                                                  
picture cartpolev1.jpg