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
    def __init__(self, policy_network, output_space, timesteps_per_batch=10000, gamma=0.99, base_line=True):
        super(MonteCarloPGRobot, self).__init__()
        self.policy_network = policy_network
        self.timesteps_per_batch = timesteps_per_batch
        self.gamma = gamma
        self.action_buffer = []
        self.state_buffer = []
        self.reward_buffer = []
        self.input_batch_memory = []
        self.action_batch_memory = []
        self.reward_batch_memory = []
        self.output_space = output_space
        self.iteration_time = 0
        self.now_timesteps = 0
        self.base_line = base_line

    def update(self, observation, reward, done):
        self.state_buffer.append(observation)
        self.action_buffer.append(self.old_action)
        self.reward_buffer.append(reward)
        if done:
            r = 0.0
            reward_list = np.zeros(len(self.state_buffer))
            for i in reversed(xrange(len(self.state_buffer))):
                r = r * self.gamma + self.reward_buffer[i]
                reward_list[i] = r
            self.input_batch_memory.extend(self.state_buffer)
            self.action_batch_memory.extend(self.action_buffer)
            self.reward_batch_memory.append(np.array(reward_list))
            self.now_timesteps += len(self.reward_buffer)
            self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

            if self.now_timesteps >= self.timesteps_per_batch:
                self.iteration_time += 1
                max_len = max(len(reward) for reward in self.reward_batch_memory)
                if self.base_line:
                    padded_rewards = [np.concatenate([reward, np.zeros(max_len - len(reward))])
                                      for reward in self.reward_batch_memory]
                    base_line = np.mean(padded_rewards, axis=0)
                    reward_list = [reward - base_line[:len(reward)]
                                   for reward in self.reward_batch_memory]
                    reward_list = np.concatenate(reward_list)
                else:
                    reward_list = np.concatenate([reward for reward in self.reward_batch_memory])
                # print (reward_list-np.mean(reward_list))/(np.std(reward_list)+1e-6).shape
                reward_ori_list = [reward[0] for reward in self.reward_batch_memory]
                print ''
                print '-----------------------------------'
                print 'Iteration {}'.format(self.iteration_time)
                print 'mean reward {}'.format(np.mean(reward_ori_list))
                print 'max reward {}'.format(np.max(reward_ori_list))
                print 'mean normal reward {}'.format(np.mean(reward_list))
                print 'max normal reward {}'.format(np.max(reward_list))
                print 'episode number {}'.format(len(self.reward_batch_memory))

                print np.array(self.input_batch_memory).shape
                print np.array(self.action_batch_memory).reshape((-1, self.output_space)).shape

                self.policy_network.train_by_grad(np.array(self.input_batch_memory),
                                                  np.array(self.action_batch_memory).reshape((-1, self.output_space)),
                                                  (reward_list - np.mean(reward_list)) / (np.std(reward_list) + 1e-6))

                self.input_batch_memory = []
                self.action_batch_memory = []
                self.reward_batch_memory = []
                self.now_timesteps = 0


class DiscreteMonteCarloPGRobot(MonteCarloPGRobot):
    def __init__(self, policy_network, output_space, timesteps_per_batch=10000, gamma=1.0, base_line=True):
        super(DiscreteMonteCarloPGRobot, self).__init__(policy_network, output_space, timesteps_per_batch, gamma,base_line)

    def _predict(self, observation):
        action_probability = self.policy_network.predict(np.array([observation]))[0]
        action = np.random.multinomial(1, action_probability)
        return action.argmax()


class ContinuousMonteCarloPGRobot(MonteCarloPGRobot):
    def __init__(self, policy_network, output_space, timesteps_per_batch=10000, gamma=0.99, base_line=True):
        super(ContinuousMonteCarloPGRobot, self).__init__(policy_network, output_space, timesteps_per_batch, gamma, base_line)

    def _predict(self, observation):
        from numpy.random import multivariate_normal
        action_probability = self.policy_network.predict(np.array([observation]))[0]
        return multivariate_normal(action_probability[:self.output_space],
                                   action_probability[self.output_space:].reshape(self.output_space,
                                                                                  self.output_space))
