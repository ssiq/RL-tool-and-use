from rl_tool.robot import Robot
import numpy as np
import lasagne
import theano
import theano.tensor as T
import copy

from DeterminePolicyNetwork import DeterminePolicyNetwork
from ContinousQValueNetwork import ContinousQValueNetwork


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
    def __init__(self, policy_network, q_value_network, random_process, memory, policy_update_method,
                 q_value_update_method, q_value_weight_decay=1e-4, policy_weight_decay=1e-4, batch_size=64, gamma=0.99, tao=0.001, ):
        super(DDPG, self).__init__()
        if not isinstance(policy_network, DeterminePolicyNetwork):
            raise ValueError('The policy network for ddpg should be the DeterminePolicyNetwork')
        if not isinstance(q_value_network, ContinousQValueNetwork):
            raise ValueError('The q value network for ddpg should be the ContinousQValueNetwork')
        self.random_process = random_process
        self.memory = memory
        self.old_state = None
        self.policy_update_method = policy_update_method
        self.q_value_update_method = q_value_update_method
        self.q_value_weight_decay = q_value_weight_decay
        self.policy_weight_decay = policy_weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.tao = tao
        self.policy_network = policy_network
        self.q_value_network = q_value_network
        self.target_policy_network = copy.deepcopy(policy_network)
        self.target_q_value_network = copy.deepcopy(q_value_network)
        self.one_episode_loss = []
        self.episode_itr = 0
        self.total_reward = 0.0
        self.time_steps = 0
        self.reward_history = []
        self.q_value_update_function, self.policy_update_function = \
            self._build_grad_function()

    def _build_grad_function(self):
        target, ob, action = T.matrices('target', 'ob', 'action')
        q_value = self.q_value_network.get_symbol_output(ob, action, deterministic=True)
        q_value_loss = lasagne.objectives.squared_error(target, q_value) \
                       + self.q_value_weight_decay * \
                         self.q_value_network.regularize_network_params(lasagne.regularization.l2)
        q_value_updates = self.q_value_update_method(q_value_loss, self.q_value_network.get_params(trainable=True))
        q_value_update_function = theano.function([ob, action, target],
                                                  q_value_loss, updates=q_value_updates, allow_input_downcast=True)

        p_action = self.policy_network.get_symbol_output(ob)
        p_q_value = self.q_value_network.get_symbol_output(ob, p_action, deterministic=True)
        p_loss = -T.mean(p_q_value) + self.policy_weight_decay * \
                                      self.policy_network.regularize_network_params(lasagne.regularization.l2)
        p_updates = self.policy_update_method(p_loss, self.policy_network.get_params(trainable=True))
        p_update_function = theano.function([ob], p_loss, updates=p_updates, allow_input_downcast=True)
        return q_value_update_function, p_update_function

    def _update_target_network(self, tao):
        original = [self.policy_network, self.q_value_network]
        target = [self.target_policy_network, self.target_q_value_network]
        for o, t in zip(original, target):
            t_values = t.get_all_param_values()
            o_values = o.get_all_param_values()
            new_t_values = [tao*ov + (1.0-tao)*tv for tv, ov in zip(t_values, o_values)]
            t.set_all_param_values(new_t_values)

    def _predict(self, observation):
        self.old_state = observation
        return self.policy_network.get_action(np.array([observation])) + self.random_process.sample()

    def _unpack_batch(self, batch):
        train_X = []
        action_list = []
        now_features_list = []
        done_list = []
        reward_list = []
        for old_features, action, now_features, reward, done in batch:
            action_list.append(action)
            now_features_list.append(now_features)
            done_list.append([done])
            reward_list.append([reward])
            train_X.append(old_features)
        train_X = np.array(train_X)
        action_list = np.array(action_list)
        now_features_list = np.array(now_features_list)
        done_list = np.array(done_list)
        reward_list = np.array(reward_list)
        return train_X, action_list, now_features_list, done_list, reward_list

    def update(self, observation, reward, done):
        self.memory.add_sample((self.old_state, self.old_action, observation, reward, done), done)
        batch = self.memory.sample(self.batch_size)
        self.total_reward += reward
        self.time_steps += 1
        if batch is not None:
            features, actions, next_features, dones, rewards = self._unpack_batch(batch)
            target_next_action = self.target_policy_network.get_output(next_features)
            target = (not done) * self.gamma * self.target_q_value_network.get_output(next_features, target_next_action)\
                     + rewards
            Qvalue_loss = self.q_value_update_function(features, actions, target)
            action = self.policy_update_function(features)
            self.policy_update_function(features, action)
            self.one_episode_loss.append(Qvalue_loss)
            self._update_target_network(self.tao)
            if done:
                self.reward_history.append(self.total_reward)
                print "Episode {} of Qvalue loss {}, reward {}, time steps {} and last 100 mean reward {}"\
                    .format(self.episode_itr,
                            np.mean(self.one_episode_loss),
                            self.total_reward,
                            self.time_steps,
                            np.mean(self.reward_history[-100:]))
                self.episode_itr += 1
                self.one_episode_loss = []
                self.total_reward = 0.0
                self.time_steps = 0




