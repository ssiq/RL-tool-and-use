import theano
import theano.tensor as T
import lasagne
import numpy as np


def rmsprop(loss, params, learning_rate=1.0, rho=0.9, epsilon=1e-9):
    updates = []
    grads = theano.grad(loss, params)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates.append((accu, accu_new))
        updates.append((param, param + (learning_rate * grad /
                                T.sqrt(accu_new + epsilon))))

    return updates


class PolicyNetwork(object):
    def __init__(self, predict_function, update_function):
        self.predict_function = predict_function
        self.update_function = update_function

    def predict(self, state):
        return self.predict_function(state)

    def train_by_grad(self, X, action, reward):
        return self.update_function(X, action, reward)

    def _build_network(self, input_var):
        '''
        It is a virtual function which used to create the network layers(lasagne)
        You should use the input_var as the input_layer's input
        For a discrete policy the network's output should be the multinomial distribution's parameter
        For a Continuous policy the network's output should be
        the guassian mean and covariance(The output's first N is mean and the last N*N output is covariance)
        :return:the network layers(lasagne)
        '''
        raise Exception('You should implement the network in this function')


class DiscretePolicyNetwork(PolicyNetwork):
    def __init__(self, input_number, output_number, optimizer, **optimizer_config):
        self.input_number = input_number
        self.output_number = output_number
        input_var = T.fmatrix('x')
        self.network = self._build_network(input_var)
        prediction = lasagne.layers.get_output(self.network)
        self.predict_function = theano.function([input_var], prediction, allow_input_downcast=True)
        all_params = lasagne.layers.get_all_params(self.network, trainable=True)
        action_list = T.ivector('action_list')
        reward_list = T.fvector('reward_list')
        N = input_var.shape[0]
        loss_function = T.log(prediction[T.arange(N), action_list]).dot(reward_list) / N
        updates = optimizer(loss=loss_function, params=all_params,
                            **optimizer_config)
        self.update_function = theano.function([input_var, action_list, reward_list],
                                               loss_function, updates=updates, allow_input_downcast=True)
        super(DiscretePolicyNetwork, self).__init__(self.predict_function, self.update_function)


class ContinuousPolicyNetwork(PolicyNetwork):
    def __init__(self, input_number, output_number, optimizer, **optimizer_config):
        self.input_number = input_number
        self.output_number = output_number
        input_var = T.fmatrix('x')
        self.network = self._build_network(input_var)
        prediction = lasagne.layers.get_output(self.network)
        self.predict_function = theano.function([input_var], prediction, allow_input_downcast=True)
        all_params = lasagne.layers.get_all_params(self.network, trainable=True)
        action_list = T.fmatrix('action_list')
        reward_list = T.fvector('reward_list')
        N = input_var.shape[0]
        mu = prediction[T.arange(N), :self.output_number]
        sigma = T.reshape(prediction[T.arange(N), self.output_number:], (N, output_number, output_number))
        action_minus_mean = T.reshape(action_list-mu, (N, 1, self.output_number))
        exper = theano.scan(lambda x, y: x.dot(y).dot(T.transpose(x, [1, 0])), sequences=[action_minus_mean, sigma])[0]
        divisor = T.sqrt((np.pi**self.output_number) * theano.scan(lambda x: T.nlinalg.det(T.nlinalg.matrix_inverse(x)), sequences=sigma)[0])

        exper = T.reshape(exper, (N,))
        loss_function = T.log(exper/divisor).dot(reward_list)/N
        updates = optimizer(loss=loss_function, params=all_params, **optimizer_config)
        update_function = theano.function([input_var, action_list, reward_list],
                                          loss_function, updates=updates, allow_input_downcast=True)
        super(ContinuousPolicyNetwork, self).__init__(self.predict_function, update_function)