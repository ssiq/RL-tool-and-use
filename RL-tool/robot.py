class Robot(object):
    def __init__(self):
        pass

    def set_action_space(self, action_space):
        pass

    def set_observation_space(self, observation_space):
        pass

    def response(self, observation):
        pass

    def update(self, observation, reward, done):
        pass


class MonteCarloRobot(Robot):
    def __init__(self, value_approximator=None):
        super(MonteCarloRobot, self).__init__()
        self.value_approximator = value_approximator

    def set_observation_space(self, observation_space):
        super(MonteCarloRobot, self).set_observation_space(observation_space)

    def update(self, observation, reward, done):
        super(MonteCarloRobot, self).update(observation, reward, done)

    def set_action_space(self, action_space):
        super(MonteCarloRobot, self).set_action_space(action_space)

    def response(self, observation):
        super(MonteCarloRobot, self).response(observation)



