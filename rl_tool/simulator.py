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

    def run(self, episode_number=20, max_episode_length=100, render_per_iterations=100, done_reward=-1):
        for i_episode in xrange(episode_number):
            observation = self.env.reset()
            self.robot.reset()
            total_reward = 0.0
            for t in xrange(max_episode_length):
                if i_episode % render_per_iterations == 0:
                    self.env.render()
                action = self.robot.response(observation)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if done or t == max_episode_length-1:
                    self.robot.update(observation, reward, True)
                    break
                else:
                    self.robot.update(observation, reward, done)

