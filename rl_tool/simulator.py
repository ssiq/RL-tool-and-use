from utility import plot_lines
import numpy as np


class Simulator(object):
    def __init__(self, env=None, robot=None, verbose=False, save_path=None):
        self.env = env
        self.robot = robot
        # self._set_env_and_robot()
        self.verbose = verbose
        self.save_path = save_path

    def _set_env_and_robot(self):
        self.robot = self.robot \
            .set_action_space(self.env.action_space) \
            .set_observation_space(self.env.observation_space)

    def set_env(self, env):
        self.env = env
        # if self.robot is not None:
        #     self._set_env_and_robot()
        return self

    def set_robot(self, robot):
        self.robot = robot
        # if self.env is not None:
        #     self._set_env_and_robot()
        return self

    def run(self, episode_number=20, max_episode_length=100, done_reward=-1, has_loss=True):
        reward_list = []
        loss_list = []
        for i_episode in xrange(episode_number):
            losses = []
            observation = self.env.reset()
            self.robot.reset()
            total_reward = 0.0
            for t in xrange(max_episode_length):
                self.env.render()
                action = self.robot.response(observation)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if done or t == max_episode_length-1:
                    losses.append(self.robot.update(observation, done_reward, True))
                    print("Episode {} finished after {} timesteps with reward {}".
                          format(i_episode, t + 1, total_reward))
                    break
                else:
                    losses.append(self.robot.update(observation, reward, done))
            reward_list.append(total_reward)
            if has_loss:
                loss_list.append(np.array(loss_list).mean())
        if self.save_path is not None:
            lines = [(xrange(1, len(reward_list)+1), reward_list)]
            labels = ['reward']
            if has_loss:
                lines.append((xrange(1, len(loss_list)+1), loss_list))
                labels.append('loss')
            plot_lines(lines, labels, self.save_path)
