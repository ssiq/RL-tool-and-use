from rl_tool.robot import Robot
import numpy as np


class PGRobot(Robot):
    def __init__(self):
        super(PGRobot, self).__init__()
        self.old_action = None

    def _predict(self, observation):
        raise Exception('You should implement the _predict function in the subclass')

    def response(self, observation):
        action = self._predict(observation)
        self.old_action = action
        return action

    def reset(self):
        super(PGRobot, self).reset()

    def update(self, observation, reward, done):
        super(PGRobot, self).update(observation, reward, done)


class MonteCarloPGRobot(PGRobot):
    def __init__(self, policy_network, output_space, update_frequency=3, gamma=0.99):
        super(MonteCarloPGRobot, self).__init__()
        self.policy_network = policy_network
        self.update_frequency = 3
        self.gamma = 0.9
        self.action_buffer = []
        self.state_buffer = []
        self.reward_buffer = []
        self.input_batch_memory = []
        self.output_batch_memory = []
        self.output_space = output_space
        self.iteration_time = 0
        self.old_probability = None

    def update(self, observation, reward, done):
        self.state_buffer.append(observation)
        self.action_buffer.append(self.old_probability)
        self.reward_buffer.append(reward)
        if done:
            r = 0.0
            self.iteration_time += 1
            for i in reversed(xrange(len(self.state_buffer))):
                r = r * self.gamma + self.reward_buffer[i]
                self.input_batch_memory.append(self.state_buffer[i])
                self.output_batch_memory.append(self.action_buffer[i] * r)
                self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

            if self.iteration_time % self.update_frequency == 0:
                self.policy_network.train_by_grad(np.array(self.input_batch_memory),
                                                  np.array(self.output_batch_memory))
                self.input_batch_memory = []
                self.output_batch_memory = []


class DiscreteMonteCarloPGRobot(MonteCarloPGRobot):
    def __init__(self, policy_network, output_space, update_frequency=3, gamma=0.99):
        super(DiscreteMonteCarloPGRobot, self).__init__(policy_network, output_space, update_frequency, gamma)

    def _predict(self, observation):
        action_probability = self.policy_network.predict(observation)[0]
        self.old_probability = action_probability
        return action_probability.argmax()

    def _calculate_grad(self, action):
        t = np.zeros(self.output_space)
        t[action] = 1.0
        return t - self.old_probability
