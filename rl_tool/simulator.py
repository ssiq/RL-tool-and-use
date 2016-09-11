class Simulator(object):
    def __init__(self, env=None, robot=None, verbose=False):
        self.env = env
        self.robot = robot
        # self._set_env_and_robot()
        self.verbose = verbose

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

    def run(self, episode_number=20, max_episode_length=100, done_reward=-1):
        for i_episode in xrange(episode_number):
            observation = self.env.reset()
            self.robot.reset()
            total_reward = 0.0
            for t in xrange(max_episode_length):
                self.env.render()
                action = self.robot.response(observation)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    self.robot.update(observation, done_reward, done)
                    print("Episode {} finished after {} timesteps with epsilon {} and reward {}".
                          format(i_episode, t + 1, self.robot.epsilon, total_reward))
                    break
                else:
                    self.robot.update(observation, reward, done)
            else:
                print("Episode {} finished after {} timesteps with epsilon {} and reward {}".
                      format(i_episode, max_episode_length, self.robot.epsilon, total_reward))
