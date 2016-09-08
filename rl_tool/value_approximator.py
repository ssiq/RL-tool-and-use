import numpy as np


class ValueApproximator(object):
    def __init__(self):
        pass

    def get_value(self, state, action):
        pass

    def update_value(self, target_value, state, action):
        pass

    def batch_update_value(self, exprience_list):
        for target_value, state, action in exprience_list:
            self.update_value(target_value, state, action)


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
        if self.itr % 10000 == 0:
            print 'targe:{}, approximate:{},delta:{}'.format(target_value, self._get_value(state, action),
                                                             target_value - self._get_value(state, action))
            self.learning_rate /= 2.0
        self.w[action, :] += self.learning_rate * (target_value - self._get_value(state, action)) * state

    def get_value(self, state, action):
        super(DiscreteLinearValueApproximator, self).get_value(state, action)
        return self._get_value(state, action)


class NetworkApproximater(ValueApproximator):
    def __init__(self, network, action_number):
        super(NetworkApproximater, self).__init__()
        self.network = network
        self.action_number = action_number

    def _get_value(self, state, action):
        return self.network.predict(np.array([state]))[0][action]

    def get_value(self, state, action):
        return self._get_value(state, action)

    def update_value(self, target_value, state, action):
        self.batch_update_value([(target_value, state, action)])

    def batch_update_value(self, exprience_list):
        train_X = []
        train_y = []
        for target_value, state, action in exprience_list:
            train_X.append(state)
            y = self.network.predict(np.array([state]))[0]
            y[action] = target_value
            train_y.append(y)
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        print 'begin to fit network'
        self.network.fit(train_X, train_y, nb_epoch=1, batch_size=len(exprience_list))
        print 'end to fit network'
