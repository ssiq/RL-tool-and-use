import theano
import theano.tensor as T
import lasagne

from Network import Network


class DeterminePolicyNetwork(Network):
    def __init__(self):
        network, self.ob_layer = self._build_network()
        super(DeterminePolicyNetwork, self).__init__(network)
        ob_var = T.matrix('ob_var')
        default_symbol_output = self.get_symbol_output(ob_var, deterministic=True)
        self.policy_function = theano.function([ob_var], default_symbol_output, allow_input_downcast=True)

    def get_output(self, observation):
        return self.policy_function(observation)

    def get_action(self, observation):
        return self.get_output(observation)[0]

    def get_symbol_output(self, observation_var, **kwargs):
        return super(DeterminePolicyNetwork, self).get_sym_output(inputs={self.ob_layer: observation_var}, **kwargs)

    def _build_network(self):
        '''

        :return:The lasagne layer and the observation input layer
        '''
        raise Exception('You should implement the _get_network function in the subclass')


class MlpDeterminePolicyNetwork(DeterminePolicyNetwork):
    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=lasagne.nonlinearities.rectify,
                 hidden_W_init=lasagne.init.HeUniform(),
                 hidden_b_init=lasagne.init.Constant(0.),
                 output_nonlinearity=lasagne.nonlinearities.tanh,
                 output_W_init=lasagne.init.Uniform(-3e-3, 3e-3),
                 output_b_init=lasagne.init.Uniform(-3e-3, 3e-3),
                 bn=False):
        '''

        :param hidden_sizes:a list contains a list of layer size from first hidden layer to the last hidden layer
        :param hidden_nonlinearity: hidden layer active function
        :param hidden_W_init: hidden layer W init function
        :param hidden_b_init: hidden layer b init function
        :param output_nonlinearity: output layer active function
        :param output_W_init: output layer W init function
        :param output_b_init: output layer b init function
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hidden_W_init = hidden_W_init
        self.hidden_b_init = hidden_b_init
        self.output_nonlinearity = output_nonlinearity
        self.output_W_init = output_W_init
        self.output_b_init = output_b_init
        self.bn = bn
        super(MlpDeterminePolicyNetwork, self).__init__()

    def _build_network(self):
        input_layer = lasagne.layers.InputLayer(shape=(None, self.input_shape))
        network = input_layer
        for size in self.hidden_sizes:
            network = lasagne.layers.DenseLayer(network,
                                                num_units=size,
                                                W=self.hidden_W_init,
                                                b=self.hidden_b_init,
                                                nonlinearity=self.hidden_nonlinearity)