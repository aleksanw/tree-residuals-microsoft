import pdb
from pdb import set_trace as bp

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import scipy.misc
import os

from collections import defaultdict
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from pylab import rcParams

class Qnetwork():
    def __init__(self, input_shape, hiddens, num_actions, dueling_type='avg', layer_norm=False):
        
        self.input =  tf.placeholder(shape= (None, np.prod(input_shape)),dtype=tf.float32)
        out = self.input
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        
#        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
#        self.conv1 = slim.conv2d( \
#            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
#        self.conv2 = slim.conv2d( \
#            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
#        self.conv3 = slim.conv2d( \
#            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
#        self.conv4 = slim.conv2d( \
#            inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
        
        #We take the output from the final hidden layer and split it into separate advantage and value streams.
        #Then combine them together to get our final Q-values.

#        self.streamAC,self.streamVC = tf.split(out,2,3)
#        self.streamA = slim.flatten(self.streamAC)
#        self.streamV = slim.flatten(self.streamVC)
#        xavier_init = tf.contrib.layers.xavier_initializer()
#        self.AW = tf.Variable(xavier_init([hidden//2,num_actions]))
#        self.VW = tf.Variable(xavier_init([hidden//2,1]))
#        self.Advantage = tf.matmul(self.streamA,self.AW)
#        self.Value = tf.matmul(self.streamV,self.VW)

        self.Advantage = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        self.Value = layers.fully_connected(out, num_outputs=1, activation_fn=None)

        # dueling_type == 'avg'
        # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
        # dueling_type == 'max'
        # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
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
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
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
    succ_epi = 0
    acc_rew = 0
    for _epi in range(1,101):
        _s = np.asarray(env.reset()).flatten()
        _d = False
        while not _d:
            _a = sess.run(q_max,feed_dict={input_ph:[_s]})[0]
            _s1,_r,_d,_ = env.step(_a)
            _s = np.asarray(_s1).flatten()
            acc_rew += _r
        succ_epi += 1 if _r == 1 else 0
    succ_epi = succ_epi*100.0/_epi

    return succ_epi, acc_rew

def run_evaluate(env, sess, q_max, input_ph, visualize=False, v_func=None):
    total_interactions = []
    total_interactions.append(total_steps)

    total_rewards = []
    _ , acc_rew = evaluate(env, sess, q_max, input_ph, visualize=False, v_func=None)
    total_rewards.append(acc_rew)

    return total_interactions, total_rewards


def run(env):
    batch_size = 3 #How many experiences to use for each training step.
    update_freq = 4 #How often to perform a training step.
    y = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0 #0.1 #Final chance of random action
    max_epLength = 100 #The max allowed length of our episode.
    load_model = False #Whether to load a saved model.
    tau = 0.001 #Rate to update target network toward primary network
    hiddens = [50, 50]

    num_actions = env.action_space.n
    input_shape = list(env.observation_space.shape)

    num_episodes = 15000 #How many episodes of game environment to train network with.
    annealing_steps = 30000. #How many steps of training to reduce startE to endE.
    pre_train_steps = 10000 #How many steps of random actions before training begins.

    succ_threshold = 100

    tf.reset_default_graph()
    mainQN = Qnetwork(input_shape, hiddens, num_actions, layer_norm=False)
    targetQN = Qnetwork(input_shape, hiddens, num_actions, layer_norm=False)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables,tau)

    myBuffer = experience_buffer()

    #Set the rate of random action decrease. 
    e = startE
    stepDrop = (startE - endE)/annealing_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0
    acc_rew = 0

    total_interactions = []
    total_rewards = []

    finish = False
    with tf.Session() as sess:
        sess.run(init)

        succ_epi = 0
        for epi in range(1, num_episodes+1):
            if finish: break
            episodeBuffer = experience_buffer()
            #Reset environment and get first new observation
            s = np.asarray(env.reset()).flatten()
            #s = processState(s)
            d = False
            rAll = 0
            j = 0
            #The Q-Network
            while (j < max_epLength) and (not d): #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                   a = np.random.randint(0,env.action_space.n)
                else:
                    #if e == 0: print(a)
                    a = sess.run(mainQN.predict,feed_dict={mainQN.input:[s]})[0]
                s1,r,d,_ = env.step(a)
                s1 = np.asarray(s1).flatten()
                total_steps += 1
                episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.

                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                        e = max(e,0)

                    #if e == 0:  env.visualize_on()

                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        #Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict,feed_dict={mainQN.input:np.vstack(trainBatch[:,3])})
                        Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.input:np.vstack(trainBatch[:,3])})
                        end_multiplier = -(trainBatch[:,4] - 1)
                        doubleQ = Q2[range(batch_size),Q1]
                        targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                        #Update the network with our target values.
                        _ = sess.run(mainQN.updateModel, \
                            feed_dict={mainQN.input:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})

                        updateTarget(targetOps,sess) #Update the target network toward the primary network.
                rAll += r
                s = s1

            if r==1: succ_epi += 1

            myBuffer.add(episodeBuffer.buffer)
            #jList.append(j)
            rList.append(rAll)

            k = 150
            if epi % k == 0:
                acc_rew += np.sum(rList)
                print(f"Episode {epi} Reward {np.mean(rList)} epsilon {e}")
                rList = []
                succ_epi = 0
                _succ_epi, _acc_rew = evaluate(env, sess, mainQN.predict, mainQN.input)

                # Save data for future plotting
                total_interactions.append(total_steps)
                total_rewards.append(_acc_rew/100.0)
                if e==0:

                    #Evaluate
                    _succ_epi, _acc_rew = evaluate(env, sess, mainQN.predict, mainQN.input)


                    if _succ_epi>=succ_threshold:
                            print("Evaluation success rate over 100 episodes: {}%; Total reward: {} ".format(_succ_epi, _acc_rew))
                            break

        print("Training succes rate: {}%".format(acc_rew*100.0/epi))

        _succ_epi, _acc_rew = evaluate(env, sess, mainQN.predict, mainQN.input, visualize=False, v_func=mainQN.Value)
        print("Evaluation success rate over 100 episodes: {}%; Total reward: {} ".format(_succ_epi, _acc_rew))

        return total_interactions, total_rewards
