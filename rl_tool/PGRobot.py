from rl_tool.robot import Robot
import numpy as np
import lasagne
import theano
import theano.tensor as T
import copy


class PGRobot(Robot):
    def __init__(self):
        super(PGRobot, self).__init__()
        self.old_action = None

    def _predict(self, observation):
        raise Exception('You should implement the _predict function in the subclass')

    def response(self, observation):
        action = self._predict(observation)
        self.old_action = action
        return action

    def reset(self):
        super(PGRobot, self).reset()

    def update(self, observation, reward, done):
        super(PGRobot, self).update(observation, reward, done)


class MonteCarloPGRobot(PGRobot):
    def __init__(self, policy_network, output_space, timesteps_per_batch=10000, gamma=0.99, base_line=True):
        super(MonteCarloPGRobot, self).__init__()
        self.policy_network = policy_network
        self.timesteps_per_batch = timesteps_per_batch
        self.gamma = gamma
        self.action_buffer = []
        self.state_buffer = []
        self.reward_buffer = []
        self.input_batch_memory = []
        self.action_batch_memory = []
        self.reward_batch_memory = []
        self.output_space = output_space
        self.iteration_time = 0
        self.now_timesteps = 0
        self.base_line = base_line

    def update(self, observation, reward, done):
        self.state_buffer.append(observation)
        self.action_buffer.append(self.old_action)
        self.reward_buffer.append(reward)
        if done:
            r = 0.0
            reward_list = np.zeros(len(self.state_buffer))
            for i in reversed(xrange(len(self.state_buffer))):
                r = r * self.gamma + self.reward_buffer[i]
                reward_list[i] = r
            self.input_batch_memory.extend(self.state_buffer)
            self.action_batch_memory.extend(self.action_buffer)
            self.reward_batch_memory.append(np.array(reward_list))
            self.now_timesteps += len(self.reward_buffer)
            self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

            if self.now_timesteps >= self.timesteps_per_batch:
                self.iteration_time += 1
                max_len = max(len(reward) for reward in self.reward_batch_memory)
                if self.base_line:
                    padded_rewards = [np.concatenate([reward, np.zeros(max_len - len(reward))])
                                      for reward in self.reward_batch_memory]
                    base_line = np.mean(padded_rewards, axis=0)
                    reward_list = [reward - base_line[:len(reward)]
                                   for reward in self.reward_batch_memory]
                    reward_list = np.concatenate(reward_list)
                else:
                    reward_list = np.concatenate([reward for reward in self.reward_batch_memory])
                # print (reward_list-np.mean(reward_list))/(np.std(reward_list)+1e-6).shape
                reward_ori_list = [reward[0] for reward in self.reward_batch_memory]
                print ''
                print '-----------------------------------'
                print 'Iteration {}'.format(self.iteration_time)
                print 'mean reward {}'.format(np.mean(reward_ori_list))
                print 'max reward {}'.format(np.max(reward_ori_list))
                print 'mean normal reward {}'.format(np.mean(reward_list))
                print 'max normal reward {}'.format(np.max(reward_list))
                print 'episode number {}'.format(len(self.reward_batch_memory))

                print np.array(self.input_batch_memory).shape
                print np.array(self.action_batch_memory).reshape((-1, self.output_space)).shape

                self.policy_network.train_by_grad(np.array(self.input_batch_memory),
                                                  np.array(self.action_batch_memory).reshape((-1, self.output_space)),
                                                  (reward_list - np.mean(reward_list)) / (np.std(reward_list) + 1e-6))

                self.input_batch_memory = []
                self.action_batch_memory = []
                self.reward_batch_memory = []
                self.now_timesteps = 0


class DiscreteMonteCarloPGRobot(MonteCarloPGRobot):
    def __init__(self, policy_network, output_space, timesteps_per_batch=10000, gamma=1.0, base_line=True):
        super(DiscreteMonteCarloPGRobot, self).__init__(policy_network, output_space, timesteps_per_batch, gamma,base_line)

    def _predict(self, observation):
        action_probability = self.policy_network.predict(np.array([observation]))[0]
        action = np.random.multinomial(1, action_probability)
        return action.argmax()


class ContinuousMonteCarloPGRobot(MonteCarloPGRobot):
    def __init__(self, policy_network, output_space, timesteps_per_batch=10000, gamma=0.99, base_line=True):
        super(ContinuousMonteCarloPGRobot, self).__init__(policy_network, output_space, timesteps_per_batch, gamma, base_line)

    def _predict(self, observation):
        from numpy.random import multivariate_normal
        action_probability = self.policy_network.predict(np.array([observation]))[0]
        return multivariate_normal(action_probability[:self.output_space],
                                   action_probability[self.output_space:].reshape(self.output_space,
                                                                                  self.output_space))


class DDPG(PGRobot):
    def __init__(self, input_shape, output_shape, random_process, memory, batch_size=32, gamma=0.99, tao=0.001, ):
        super(DDPG, self).__init__()
        self.random_process = random_process
        self.memory = memory
        self.old_state = None
        self.batch_size = batch_size
        self.gamma = gamma
        self.tao = tao
        self.state_input_var, self.action_input_var = T.matrices('policy_input', 'action_input')
        self.policy_network, self.Qvalue_network = \
            self._build_network(input_shape, output_shape, self.state_input_var, self.action_input_var)
        self.target_policy_network = copy.deepcopy(self.policy_network)
        self.target_Qvalue_network = copy.deepcopy(self.Qvalue_network)
        self.input_shape = input_shape
        self.output_shape = output_shape
        build_predict = lambda x: [(self._build_predict_function(p, self.state_input_var),
                                   self._build_predict_function(q, self.action_input_var)) for (p, q) in x]
        (self.policy_predict_function, self.Qvalue_predict_function), \
        (self.target_policy_predict_function, self.target_Qvalue_predict_function) = \
            build_predict([(self.policy_network, self.Qvalue_network), (self.target_policy_network, self.target_Qvalue_network)])
        self.Qvalue_update_function, self.policy_update_function = \
            self._build_grad_function(self.state_input_var, self.action_input_var)

    def _build_grad_function(self, state_input_var, action_input_var):
        target = T.matrix('target')
        Qvalue_predict = lasagne.layers.get_output(self.Qvalue_network)
        Qvalue_loss = lasagne.objectives.squared_error(target, Qvalue_predict)
        Qvalue_loss = Qvalue_loss.mean()
        Qvalue_updates = lasagne.updates.adam(Qvalue_loss, lasagne.layers.get_all_layers(self.Qvalue_network), )
        Qvalue_update_function = theano.function([state_input_var, action_input_var, target], Qvalue_loss,
                                                 updates=Qvalue_updates, allow_input_downcast=True)
        N = state_input_var.shape[0]
        policy_predict = lasagne.layers.get_output(self.policy_network)
        dQvalue_a = theano.grad(T.sum(Qvalue_predict), action_input_var)
        policy_loss = -T.sum(dQvalue_a * policy_predict)/N
        policy_params = lasagne.layers.get_all_params(self.policy_network)
        policy_loss = theano.grad(policy_loss, policy_params)
        policy_updates = lasagne.updates.adam(policy_loss, policy_params)
        policy_update_function = theano.function([state_input_var, action_input_var], policy_loss,
                                                 updates=policy_updates, allow_input_downcast=True)
        return Qvalue_update_function, policy_update_function

    def _update_target_network(self, tao):
        original = [self.policy_network, self.Qvalue_network]
        target = [self.target_policy_network, self.target_Qvalue_network]
        for o, t in zip(original, target):
            t_values = lasagne.layers.get_all_param_values(t)
            o_values = lasagne.layers.get_all_param_values(o)
            new_t_values = [tao*ov + (1-tao)*tv for tv, ov in zip(t_values, o_values)]
            lasagne.layers.set_all_param_values(t, new_t_values)

    def _build_predict_function(self, network, input_var):
        prediction = lasagne.layers.get_output(network)
        return theano.function(input_var, prediction, allow_input_downcast=True)

    def _build_network(self, input_shape, output_shape, state_input_var, action_input_var):
        '''
        :param num_input: the state shape
        :param num_output: the output shape
        :param state_input_var: the state variable
        :param action_input_var: the action var
        :return: a determine policy network, a value network which input is combines by state and action variable
        '''
        raise Exception('You should implement the _get_network function in the subclass')

    def _predict(self, observation):
        self.old_state = observation
        return self.policy_predict_function(np.array([observation]))[0] + self.random_process.sample()

    def _unpack_batch(self, batch):
        train_X = []
        action_list = []
        now_features_list = []
        done_list = []
        reward_list = []
        for old_features, action, now_features, reward, done in batch:
            action_list.append(action)
            now_features_list.append(now_features)
            done_list.append(done)
            reward_list.append(reward)
            train_X.append(old_features)
        train_X = np.array(train_X)
        action_list = np.array(action_list)
        now_features_list = np.array(now_features_list)
        done_list = np.array(done_list)
        reward_list = np.array(reward_list)
        return train_X, action_list, now_features_list, done_list, reward_list

    def update(self, observation, reward, done):
        self.memory.add_sample((self.old_state, self.old_action, observation, reward, done))
        batch = self.memory.sample(self.batch_size)
        if batch is not None:
            features, actions, next_features, dones, rewards = self._unpack_batch(batch)
            target_next_action = self.target_policy_predict_function(next_features)
            target = self.gamma * self.target_Qvalue_predict_function(next_features, target_next_action)+\
                     (not done)*rewards
            Qvalue_loss = self.Qvalue_update_function(features, actions, target)
            action = self.policy_predict_function(features)
            self.policy_update_function(features, action)




