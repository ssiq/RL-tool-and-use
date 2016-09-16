from robot import Robot


class CERobot(Robot):
    '''The abstract base class of the CERobot

    # Argument
        try_number: the number of episode to approximate the score of now distribution
        ru: the ru is the ratio of now sampled parameters should be contained

    # Note
        This class does not implement the three functions which is related to the distribution
        and the approximate function which should be implemented in the subclass
    '''
    def __init__(self, try_number, ru):
        super(CERobot, self).__init__()
        self.try_number = try_number
        self.ru = ru
        self.now_number = 0
        self.now_score = 0
        self.weight_score_list = []
        self.now_weight = self._sample_weight()

    def _sample_weight(self):
        '''
        :return: return a weight sampled from now distribution
        '''
        raise Exception('You should implement the _sample_weight function in the subclass')

    def _get_action(self, observation, weight):
        '''
        :param observation: now observation
        :param weight: the weight used to choose action
        :return: return a action decided by now approximate function
        '''
        raise Exception('You should implement the _get_action function in the subclass')

    def _update_parameter(self, weight_score_list):
        '''
        :param weight_score_list: a list of (weight, score) that should used to update the distribution
        :return: None
        '''
        raise Exception('You should implement the _update_parameter function in the subclass')

    def update(self, observation, reward, done):
        self.now_score += reward
        if done:
            self.now_number += 1
            self.weight_score_list.append((self.now_weight, self.now_number))
            self.now_weight = self._sample_weight()
            self.now_score = 0

        if self.now_number == self.try_number:
            left_number = int(self.try_number * self.ru)
            ws_l = sorted(self.weight_score_list, lambda x, y: cmp(x[1], y[1]))[:-left_number]
            self._update_parameter(ws_l)
            self.weight_score_list = []
            self.now_weight = self._sample_weight()
            self.now_number = 0

    def response(self, observation):
        return self._get_action(observation, self.now_weight)

    def reset(self):
        super(CERobot, self).reset()
