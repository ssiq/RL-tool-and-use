import numpy as np


class ValueApproximator(object):
    def __init__(self):
        pass

    def get_value(self, state, action):
        pass

    def update_value(self, target_value, state, action):
        pass


class TableValueApproximator(ValueApproximator):
    def __init__(self, learning_rate):
        super(TableValueApproximator, self).__init__()
        self.table = {}
        self.learning_rate = learning_rate

    def _get_value(self, state, action):
        state = (state, action)
        if state in self.table:
            return self.table[state]
        else:
            return 0

    def _set_value(self, state, action, value):
        key = (state, action)
        self.table[key] = value

    def update_value(self, target_value, state, action):
        super(TableValueApproximator, self).update_value(target_value, state, action)

        new_value = self._get_value(state, action) + \
                    self.learning_rate * (target_value - self._get_value(state, action))
        self._set_value(state, action, new_value)

    def get_value(self, state, action):
        super(TableValueApproximator, self).get_value(state, action)
        return self._get_value(state, action)


class DiscreteLinearValueApproximator(ValueApproximator):
    def __init__(self, learning_rate, shape):
        super(DiscreteLinearValueApproximator, self).__init__()
        self.shape = shape
        self.w = np.random.randn(*shape)
        self.learning_rate = learning_rate
        self.itr = 0

    def _get_value(self, state, action):
        return self.w[action, :].dot(state)

    def update_value(self, target_value, state, action):
        super(DiscreteLinearValueApproximator, self).update_value(target_value, state, action)
        self.itr += 1
        if self.itr%1000 == 0:
            print 'targe:{}, approximate:{},delta:{}'.format(target_value, self._get_value(state, action),
                                                             target_value - self._get_value(state, action))
            # self.learning_rate /= 2.0
        self.w[action, :] -= self.learning_rate * (target_value - self._get_value(state, action)) * state

    def get_value(self, state, action):
        super(DiscreteLinearValueApproximator, self).get_value(state, action)
        return self._get_value(state, action)

