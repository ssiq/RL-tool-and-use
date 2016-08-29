import numpy as np


class ValueApproximator(object):
    def __init__(self):
        pass

    def get_value(self, state):
        pass

    def update_value(self, target_value, state):
        pass


class TableValueApproximator(ValueApproximator):
    def __init__(self, learning_rate):
        super(TableValueApproximator, self).__init__()
        self.table = {}
        self.learning_rate = learning_rate

    def _get_value(self, state):
        if state in self.table:
            return self.table[state]
        else:
            return 0

    def _set_value(self, state, action, value):
        key = (state, action)
        self.table[key] = value

    def update_value(self, target_value, state):
        super(TableValueApproximator, self).update_value(target_value, state)
        new_value = self._get_value(state) + \
                    self.learning_rate * (target_value - self._get_value(state))
        self._set_value(state, new_value)

    def get_value(self, state):
        super(TableValueApproximator, self).get_value(state)
        return self._get_value(state)


class LinearValueApproximator(ValueApproximator):
    def __init__(self, learning_rate, shape):
        super(LinearValueApproximator, self).__init__()
        self.shape = shape
        self.w = np.random.randn(*shape)
        self.learning_rate = learning_rate

    def _get_value(self, state):
        assert state.shape != self.shape, 'state shape not consistent'
        return np.sum(self.w * state)

    def update_value(self, target_value, state):
        super(LinearValueApproximator, self).update_value(target_value, state)
        self.w = self.learning_rate * (target_value - state) * self.w

    def get_value(self, state):
        super(LinearValueApproximator, self).get_value(state)
        return self._get_value(state)

