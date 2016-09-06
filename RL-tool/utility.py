class ApproximatorAndFearureBuilder(object):
    def __init__(self, observation_space=None, action_space=None):
        self.observation_space = observation_space
        self.action_space = action_space

    def set_observation_space(self, observation_space):
        self.observation_space = observation_space
        return self

    def set_action_space(self, action_space):
        self.action_space = action_space
        return self

    def build(self):
        pass


def flip_coin(p):
    import random
    return random.random() < p
