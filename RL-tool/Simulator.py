class Simulator(object):
    def __init__(self, env=None, robot=None, verbose=False):
        self.env = env
        self.robot = robot
        self._set_env_and_robot()
        self.verbose = verbose

    def _set_env_and_robot(self):
        self.robot = self.robot \
            .setActionSpace(self.env.action_space) \
            .setObservationSpace(self.env.observation_space)

    def set_env(self, env):
        self.env = env
        if self.robot is not None:
            self._set_env_and_robot()
        return self

    def set_robot(self, robot):
        self.robot = robot
        if self.env is not None:
            self._set_env_and_robot()
        return self

    def run(self, episode_number=20, max_episode_length=100):
        for i_episode in xrange(episode_number):
            observation = self.env.reset()
            for t in xrange(max_episode_length):
                action = self.robot.response(observation)
                observation, reward, done, info = self.env.step(action)
                self.robot.update(observation, reward, done)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
