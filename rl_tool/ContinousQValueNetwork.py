import theano
import theano.tensor as T

from Network import Network


class ContinousQValueNetwork(Network):
    def __init__(self):
        network, self.ob_layer, self.action_layer = self._build_network()
        super(ContinousQValueNetwork, self).__init__(network)
        ob_var, action_var = T.matrices('ob', 'action')
        self.q_value_function = theano.\
            function([ob_var, action_var], self.get_symbol_output(ob_var, action_var, deterministic=True),
                     allow_input_downcast=True)

    def get_output(self, observation, action):
        return self.q_value_function(observation, action)

    def get_q_value(self, observation, action):
        return self.get_output(observation, action)[0]

    def get_symbol_output(self, observation_var, action_var, **kwargs):
        return super(ContinousQValueNetwork, self).\
            get_sym_output(
            inputs={self.ob_layer: observation_var, self.action_layer: action_var}, **kwargs)

    def _build_network(self):
        '''
        :return: a q_value network, a ob input layer and a action input layer
        '''
        raise Exception('You should implement the _get_network function in the subclass')