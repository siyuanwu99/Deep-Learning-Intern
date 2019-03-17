import numpy as np 
import pandas as pd 
import tensorflow as tf 

class DQN:
    def __init__(
        self,
        n_actions,
        n_states,
        learning_rate = 0.01,
        reward_decay = 0.3,
        e_greedy = 0.9,
        batch_size = 32,
        replace_target_iteration = 10,
        memory_size = 400,
        e_greedy_increment = None,
        is_output_graph = False
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.rd = reward_decay
        self.epsilon_max = e_greedy
        self.batch_size = batch_size
        self.replace_target_iteration = replace_target_iteration
        self.memory_size = memory_size
        self.e_greedy_increment = e_greedy_increment
        self.e_greedy = 0 if e_greedy_increment is not None else self.epsilon_max

        self.q_target = []
        # learning step counter
        self.learn_step_counter = 0

        # initialize memory
        self.memory = np.zeros((self.memory_size, 2 * self.n_states + 2 ))  # [observation, action, reward, obseration_]

        # build network
        self._build_net()
        
        # parameters replacement
        target_paras = tf.get_collection('target_paras')
        evalue_paras = tf.get_collection('eval_paras')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(target_paras, evalue_paras)]

        self.sess = tf.Session()

        if is_output_graph:
            # enter 'tensorboard --logdir log' in your terminal, and then open the URL 
            tf.summary.FileWriter("logs/", graph= self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.cost_history = []


    def _build_net(self):

        # create placeholders
        self.state = tf.placeholder(tf.float32, shape=[None, self.n_states], name='state')
        self.action = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='action')
        self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='q_target')

        w_initial = tf.initializers.random_normal(0.1)
        b_initial = tf.initializers.random_normal(0.3)
        n_layer1 = 10
        # n_layer-2 = 5
        # ------------ evalue_net ------------------
        with tf.variable_scope("evalue_net"):
            
            # create collection
            c_name = ['eval_paras', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.name_scope("layer-1"):
                w1 = tf.get_variable('w1', [self.n_states, n_layer1], initializer= w_initial, collections=c_name)
                b1 = tf.get_variable('b1', [1, n_layer1], initializer= b_initial, collections=c_name)
                
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.name_scope("layer-2"):
                w2 = tf.get_variable('w2', [n_layer1, self.n_actions], initializer= w_initial, collections=c_name)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer= b_initial, collections=c_name)

                self.q_val = tf.nn.relu(tf.matmul(l1, w2) + b2)

        # ------------ target_net ------------------
        self.state_ = tf.placeholder(tf.float32, shape=[None, self.n_states], name='state_')

        with tf.variable_scope("target-net"):

            # create collection
            c_name = ['eval_paras', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.name_scope("layer-1"):
                w1 = tf.get_variable('w1', [self.n_states, n_layer1], initializer= w_initial, collections=c_name)
                b1 = tf.get_variable('b1', [1, n_layer1], initializer= b_initial, collections=c_name)

                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.name_scope("layer-2"):
                w2 = tf.get_variable('w2', [n_layer1, self.n_actions], initializer= w_initial, collections=c_name)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer= b_initial, collections=c_name)

                self.q_predict = tf.nn.relu(tf.matmul(l1, w2) + b2)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_sum(tf.losses.mean_squared_error(self.q_target, self.q_val))
        with tf.variable_scope('train'):
            self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
           


    def _store_transition(self, state, action, reward, state_):

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((state,[action, reward],state_))
         # why could this expression useful?

        index = self.memory_counter % self.memory_size

        self.memory[index, :] = transition

        self.memory_counter += 1

    
    def _choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.e_greedy:
            action_values = self.sess.run(self.q_val,
             feed_dict={self.state: observation})
            action = np.argmax(action_values)
           
        else:
            action = int(np.random.rand()*3)
            
        return action


    def learn(self):
        # renew parameters in target network
        if self.learn_step_counter % self.replace_target_iteration == 0:
            self.sess.run(self.replace_target_op)
            print(">>> REPLACE TARGET NETWORK")
        
        # select memory
        if self.memory_counter > self.memory_size:
            batch_index = np.random.choice(self.memory_size, size= self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_counter, size= self.batch_size)
        batch = self.memory[batch_index, :]

        # acquire Q-predicted and Q-evaluated
        q_predicted, q_evaluated = self.sess.run(
            [self.q_predict, self.q_val], 
            feed_dict={
                self.state: batch[:, :self.n_states],
                self.state_: batch[:, -self.n_states:]
            }
        )

        # create q_target due to backprop
        q_target = q_evaluated.copy()  # create a new array
        batch_index = np.arange(self.batch_size, dtype= np.int32)  # 
        eval_act_index = batch[:, self.n_states].astype(int)  # get action index
        reward = batch[:, self.n_states + 1]  # get reward

        q_target[batch_index, eval_act_index] = reward + self.rd * np.max(q_predicted)

        # train
        _, self.cost = self.sess.run(
            [self.train, self.loss],
            feed_dict={
                self.state: batch[:, :self.n_states],
                self.q_target: q_target
            }
        )

        self.cost_history.append(self.cost)

        if self.e_greedy_increment != None:
            self.e_greedy = self.e_greedy + self.e_greedy_increment if self.e_greedy < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1



    def view_graph(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()