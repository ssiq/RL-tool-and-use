from feature import IdentityFeatureExtractor


class Robot(object):
    def __init__(self):
        pass

    def response(self, observation):
        pass

    def update(self, observation, reward, done):
        pass


class MonteCarloRobot(Robot):
    def __init__(self, value_approximator=None):
        super(MonteCarloRobot, self).__init__()
        self.value_approximator = value_approximator

    def update(self, observation, reward, done):
        super(MonteCarloRobot, self).update(observation, reward, done)

    def response(self, observation):
        super(MonteCarloRobot, self).response(observation)


class TDRobot(Robot):
    def __init__(self, lamda=0, value_approximator=None, feature=IdentityFeatureExtractor()):
        super(TDRobot, self).__init__()
        self.lamda = 0
        self.value_approximator = value_approximator
        self.feature = feature

    def update(self, observation, reward, done):
        super(TDRobot, self).update(observation, reward, done)

    def response(self, observation):
        super(TDRobot, self).response(observation)





