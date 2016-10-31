import lasagne


class Network(object):
    def __init__(self, network):
        self.network = network

    def get_params(self, **kwargs):
        return lasagne.layers.get_all_params(self.network, **kwargs)

    def get_sym_output(self, inputs, **kwargs):
        return lasagne.layers.get_output(self.network, inputs, **kwargs)

    def get_all_param_values(self, **kwargs):
        return lasagne.layers.get_all_param_values(self.network, **kwargs)

    def set_all_param_values(self, values, **kwargs):
        return lasagne.layers.set_all_param_values(self.network, values, **kwargs)

    def regularize_network_params(self, penalty, tags={'regularizable': True}, **kwargs):
        return lasagne.regularization.regularize_network_params(self.network, penalty, tags=tags, **kwargs)