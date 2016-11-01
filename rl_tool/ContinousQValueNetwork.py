import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne.init as I
import lasagne.nonlinearities as NL
import lasagne.regularization as R

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


class MlpContinousQValueNetwork(ContinousQValueNetwork):
    def __init__(self,
                 input_shape,
                 action_shape,
                 hidden_sizes=(32, 32),
                 action_merge_layer=-2,
                 hidden_nonlinearity=NL.rectify,
                 hidden_W_init=I.HeUniform(),
                 hidden_b_init=I.Constant(0.),
                 output_nonlinearity=NL.tanh,
                 output_W_init=I.Uniform(-3e-3, 3e-3),
                 output_b_init=I.Uniform(-3e-3, 3e-3),
                 bn=False
                 ):
        '''

        :param input_shape: the input size
        :param action_shape: the action size
        :param hidden_sizes:  a hidden size list
        :param action_merge_layer:  action merge layer index
        :param hidden_nonlinearity: hidden layer active function
        :param hidden_W_init: hidden layer W init function
        :param hidden_b_init: hidden layer b init function
        :param output_nonlinearity: output layer active function
        :param output_W_init: output layer W init function
        :param output_b_init: output layer b init function
        :param bn: whether use batch norm
        '''
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.action_merge_layer = action_merge_layer
        self.hidden_W_init = hidden_W_init
        self.hidden_b_init = hidden_b_init
        self.output_nonlinearity = output_nonlinearity
        self.output_W_init = output_W_init
        self.output_b_init = output_b_init
        self.bn = bn
        super(MlpContinousQValueNetwork, self).__init__()

    def _build_network(self):
        ob_input_layer = L.InputLayer(shape=(None, self.input_shape), name='observation input layer')
        action_input_layer = L.InputLayer(shape=(None, self.action_shape), name='action layer')
        network = ob_input_layer
        if self.action_merge_layer < 0:
            self.action_merge_layer = 0
        size = len(self.hidden_sizes) + 1
        self.action_merge_layer %= size

        for idx, layer_size in enumerate(self.hidden_sizes):
            if self.bn:
                network = L.batch_norm(network)

            if idx == self.action_merge_layer:
                network = L.ConcatLayer([network, action_input_layer])

            network = L.DenseLayer(network,
                                   num_units=layer_size,
                                   W=self.hidden_W_init,
                                   b=self.hidden_b_init,
                                   nonlinearity=self.hidden_nonlinearity,
                                   name='hidden_layer_{}'.format(idx))

        if self.action_merge_layer == size - 1:
            network = L.ConcatLayer([network, action_input_layer])

        network = L.DenseLayer(network,
                               num_units=1,
                               W=self.output_W_init,
                               b=self.output_b_init,
                               nonlinearity=self.output_nonlinearity,
                               name='output_layer')

        return network, ob_input_layer, action_input_layer


