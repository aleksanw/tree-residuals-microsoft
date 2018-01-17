import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import scipy.misc

import pylab
from pylab import rcParams


class Qnetwork():
    def __init__(self, input_shape, hiddens, num_actions, dueling_type='avg', layer_norm=False):
        self.input = tf.placeholder(shape=(None, np.prod(input_shape)), dtype=tf.float32)
        self.learning_rate = 0.0001

        out = self.input
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)

        self.Advantage = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        self.Value = layers.fully_connected(out, num_outputs=1, activation_fn=None)

        if dueling_type == 'avg':
            self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        elif dueling_type == 'max':
            self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_max(self.Advantage,axis=1,keep_dims=True))
        else:
            assert False, "dueling_type must be one of {'avg','max'}"

        self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,num_actions,dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


def updateTargetGraph(tfVars,tau):
    print(tfVars)
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        #print(idx,var, tfVars[idx+total_vars//2].name)
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


def evaluate(env, sess, q_max, input_ph, visualize=False, v_func=None):
    acc_rew = 0
    rollout_count = 1
    for _epi in range(rollout_count):
        _s = np.asarray(env.reset()).flatten()
        _d = False
        while not _d:
            _a = sess.run(q_max,feed_dict={input_ph:[_s]})[0]
            _s1,_r,_d,_ = env.step(_a)
            _s = np.asarray(_s1).flatten()
            acc_rew += _r

    mean_reward = acc_rew/rollout_count
    return mean_reward


def run(env):
    num_episodes = 15000 #How many episodes of game environment to train network with.
    initial_epsilon = 1 #Starting chance of random action

    rollout_batch_size = 100 #How many experiences to use for each training step.
    update_freq = 4 #How often to perform a training step.
    y = .99 #Discount factor on the target Q-values
    end_epsilon = 0 #0.1 #Final chance of random action
    tau = 0.001 #Rate to update target network toward primary network
    hiddens = [50, 50]
    annealing_steps = 30000. #How many steps of training to reduce initial_epsilon to end_epsilon.
    pre_train_steps = 10000 #How many steps of random actions before training begins.


    succ_threshold = 100
    params = locals()
    def gen():
        load_model = False #Whether to load a saved model.

        num_actions = env.action_space.n
        input_shape = list(env.observation_space.shape)


        tf.reset_default_graph()
        mainQN = Qnetwork(input_shape, hiddens, num_actions, layer_norm=False)
        params['initial_learning_rate'] = mainQN.learning_rate
        targetQN = Qnetwork(input_shape, hiddens, num_actions, layer_norm=False)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        trainables = tf.trainable_variables()
        targetOps = updateTargetGraph(trainables,tau)
        myBuffer = experience_buffer()
        #Set the rate of random action decrease. 
        epsilon = initial_epsilon
        stepDrop = (initial_epsilon - end_epsilon)/annealing_steps
        total_steps = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(init)

            for epi in range(num_episodes):
                if epi % 5 == 0:
                    mean_reward = evaluate(env, sess, mainQN.predict, mainQN.input)
                    print(f"Episode {epi} Reward {mean_reward} epsilon {epsilon}")
                    yield total_steps, mean_reward

                episodeBuffer = experience_buffer()
                #Reset environment and get first new observation
                s = np.asarray(env.reset()).flatten()
                done = False
                #The Q-Network
                while not done:
                    #Choose an action by greedily (with epsilon chance of random action) from the Q-network
                    if np.random.rand(1) < epsilon or total_steps < pre_train_steps:
                       a = np.random.randint(0,env.action_space.n)
                    else:
                        a = sess.run(mainQN.predict,feed_dict={mainQN.input:[s]})[0]
                    s1,r,done,_ = env.step(a)
                    s1 = np.asarray(s1).flatten()
                    total_steps += 1
                    episodeBuffer.add(np.reshape(np.array([s,a,r,s1,done]),[1,5])) #Save the experience to our episode buffer.

                    if total_steps > pre_train_steps:
                        if epsilon > end_epsilon:
                            epsilon -= stepDrop
                            epsilon = max(epsilon,0)

                        if total_steps % (update_freq) == 0:
                            trainBatch = myBuffer.sample(rollout_batch_size) #Get a random batch of experiences.
                            #Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict,feed_dict={mainQN.input:np.vstack(trainBatch[:,3])})
                            Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.input:np.vstack(trainBatch[:,3])})
                            end_multiplier = -(trainBatch[:,4] - 1)
                            doubleQ = Q2[range(rollout_batch_size),Q1]
                            targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                            #Update the network with our target values.
                            _ = sess.run(mainQN.updateModel, \
                                feed_dict={mainQN.input:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})

                            updateTarget(targetOps,sess) #Update the target network toward the primary network.
                    s = s1

                myBuffer.add(episodeBuffer.buffer)
    return (params, gen())
