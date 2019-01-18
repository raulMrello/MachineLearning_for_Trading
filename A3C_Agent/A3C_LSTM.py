#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################################
####################################################################################
# A3C-LSTM implementation based on A3C developed by:
#   https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
####################################################################################
####################################################################################


####################################################################################
# Required libraries

import numpy as np
import tensorflow as tf
import gym, time, random, threading
from keras.models import *
from keras.layers import *
from keras import backend as K

####################################################################################
####################################################################################
# Class Brain
####################################################################################
####################################################################################
class Brain:
  train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
  lock_queue = threading.Lock()
  #------------------------------------------------------------------------------
  def __init__(self, lbw, num_state, num_actions, loss_v, loss_entropy, min_batch, gamma_n, learning_rate, none_state):
    self.session = tf.Session()
    K.set_session(self.session)
    K.manual_variable_initialization(True)
    self.lbw = lbw
    self.num_state = num_state
    self.num_actions = num_actions
    self.loss_v = loss_v
    self.loss_entropy = loss_entropy
    self.min_batch = min_batch
    self.gamma_n=gamma_n
    self.learning_rate = learning_rate
    self.none_state=none_state
    self.model = self._build_model()
    self.graph = self._build_graph(self.model)
    self.session.run(tf.global_variables_initializer())
    self.default_graph = tf.get_default_graph()
    self.default_graph.finalize()	# avoid modifications

  #------------------------------------------------------------------------------
  def _build_model(self):
    """ Build the A3C-LSTM neural net """
    l_input = Input(shape=(self.lbw, self.num_state))
    l_lstm1 = LSTM(64, return_sequences=True)(l_input)
    l_lstm2 = LSTM(64, return_sequences=True)(l_lstm1)
    l_lstm3 = LSTM(64, return_sequences=False)(l_lstm2)
    l_dense = Dense(128, activation='linear')(l_lstm3)
    out_actions = Dense(self.num_actions, activation='softmax')(l_dense)
    out_value   = Dense(1, activation='linear')(l_dense)
    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    model._make_predict_function()	# have to initialize before threading
    return model

  #------------------------------------------------------------------------------
  def _build_graph(self, model):
    s_t = tf.placeholder(tf.float32, shape=(None, self.num_state))
    a_t = tf.placeholder(tf.float32, shape=(None, self.num_actions))
    r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward		
    p, v = model(s_t)
    log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
    advantage = r_t - v
    loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
    loss_value  = self.loss_v * tf.square(advantage)												# minimize value error
    entropy = self.loss_entropy * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)
    loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
    optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=.99)
    minimize = optimizer.minimize(loss_total)
    return s_t, a_t, r_t, minimize

  #------------------------------------------------------------------------------
  def optimize(self):
    if len(self.train_queue[0]) < self.min_batch:
      time.sleep(0)	# yield
      return
    with self.lock_queue:
      if len(self.train_queue[0]) < self.min_batch:	# more thread could have passed without lock
        return 									# we can't yield inside lock
      s, a, r, s_, s_mask = self.train_queue
      self.train_queue = [ [], [], [], [], [] ]
    s = np.vstack(s)
    a = np.vstack(a)
    r = np.vstack(r)
    s_ = np.vstack(s_)
    s_mask = np.vstack(s_mask)
    if len(s) > 5*self.min_batch: print("Optimizer alert! Minimizing batch of %d" % len(s))
    v = self.predict_v(s_)
    r = r + self.gamma_n * v * s_mask	# set v to 0 where s_ is terminal state
    s_t, a_t, r_t, minimize = self.graph
    self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

  #------------------------------------------------------------------------------
  def train_push(self, s, a, r, s_):
    with self.lock_queue:
      self.train_queue[0].append(s)
      self.train_queue[1].append(a)
      self.train_queue[2].append(r)
      if s_ is None:
        self.train_queue[3].append(self.none_state)
        self.train_queue[4].append(0.)
      else:	
        self.train_queue[3].append(s_)
        self.train_queue[4].append(1.)

  #------------------------------------------------------------------------------
  def predict(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)
      return p, v

  #------------------------------------------------------------------------------
  def predict_p(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)		
      return p

  #------------------------------------------------------------------------------
  def predict_v(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)		
      return v



####################################################################################
####################################################################################
# Class Agent
####################################################################################
####################################################################################
class Agent:

  #------------------------------------------------------------------------------
  def __init__(self, eps_start, eps_end, eps_steps, num_actions, gamma, gamma_n, n_step_return, frames=0):
    self.eps_start = eps_start
    self.eps_end   = eps_end
    self.eps_steps = eps_steps
    self.num_actions = num_actions
    self.gamma = gamma
    self.gamma_n = gamma_n
    self.n_step_return = n_step_return
    self.frames = frames

    self.memory = []	# used for n_step return
    self.R = 0.

  #------------------------------------------------------------------------------
  def getEpsilon(self):
    if(self.frames >= self.eps_steps):
      return self.eps_end
    else:
      return self.eps_start + self.frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

  #------------------------------------------------------------------------------
  def act(self, s):
    eps = self.getEpsilon()			
    self.frames = self.frames + 1
    if random.random() < eps:
      return random.randint(0, self.num_actions-1)
    else:
      s = np.array([s])
      p = brain.predict_p(s)[0]
      # a = np.argmax(p)
      a = np.random.choice(self.num_actions, p=p)
      return a
	
  #------------------------------------------------------------------------------
  def train(self, s, a, r, s_):
    def get_sample(memory, n):
      s, a, _, _  = memory[0]
      _, _, _, s_ = memory[n-1]
      return s, a, self.R, s_
    a_cats = np.zeros(self.num_actions)	# turn action into one-hot representation
    a_cats[a] = 1 
    self.memory.append( (s, a_cats, r, s_) )
    self.R = ( self.R + r * self.gamma_n ) / self.gamma
    if s_ is None:
      while len(self.memory) > 0:
        n = len(self.memory)
        s, a, r, s_ = get_sample(self.memory, n)
        brain.train_push(s, a, r, s_)
        self.R = ( self.R - self.memory[0][2] ) / self.gamma
        self.memory.pop(0)		
      self.R = 0
    if len(self.memory) >= self.n_step_return:
      s, a, r, s_ = get_sample(self.memory, self.n_step_return)
      brain.train_push(s, a, r, s_)
      self.R = self.R - self.memory[0][2]
      self.memory.pop(0)	
  # possible edge case - if an episode ends in <N steps, the computation is incorrect
    

####################################################################################
####################################################################################
# Class Environment
####################################################################################
####################################################################################
class Environment(threading.Thread):
  stop_signal = False

  #------------------------------------------------------------------------------
  def __init__( self, 
                env, 
                eps_start, 
                eps_end, 
                eps_steps, 
                num_state, 
                num_actions,
                gamma,
                gamma_n,
                n_step_return):

    threading.Thread.__init__(self)
    self.render = False
    self.env = env
    self.agent = Agent(eps_start, eps_end, eps_steps, num_actions, gamma, gamma_n, n_step_return)

  #------------------------------------------------------------------------------
  def runEpisode(self):
    s = self.env.reset()
    R = 0
    while True:         
      time.sleep(THREAD_DELAY) # yield 
      if self.render: self.env.render()
      a = self.agent.act(s)
      s_, r, done, info = self.env.step(a)
      if done: # terminal state
        s_ = None
      self.agent.train(s, a, r, s_)
      s = s_
      R += r
      if done or self.stop_signal:
        break
    print("Total R:", R)

  #------------------------------------------------------------------------------
  def run(self):
    while not self.stop_signal:
      self.runEpisode()

  #------------------------------------------------------------------------------
  def stop(self):
    self.stop_signal = True


####################################################################################
####################################################################################
# Class Optimizer
####################################################################################
####################################################################################
class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			brain.optimize()

	def stop(self):
		self.stop_signal = True


# ####################################################################################
# ####################################################################################
# # Test Execution
# ####################################################################################
# ####################################################################################

# ####################################################################################
# # Global Constants:

# # Gym environment name
# ENV = 'TradeSim-v0'

# # Execution time in seconds
# RUN_TIME = 30
# # Parallel threads
# THREADS = 8
# # Num optimizers
# OPTIMIZERS = 2
# # Delays between threads
# THREAD_DELAY = 0.001

# # A3C learning hyper-parameters
# GAMMA = 0.99
# N_STEP_RETURN = 8
# GAMMA_N = GAMMA ** N_STEP_RETURN
# EPS_START = 0.4
# EPS_STOP  = .15
# EPS_STEPS = 75000
# MIN_BATCH = 32
# LEARNING_RATE = 5e-3
# LOSS_V = .5			# v loss coefficient
# LOSS_ENTROPY = .01 	# entropy coefficient



# env_test = Environment( env, 
#                         eps_start=0., 
#                         eps_end=0., 
#                          eps_steps=EPS_STEPS, 
#                          num_state=NUM_STATE, 
#                          num_actions=NUM_ACTIONS,
#                          gamma = GAMMA,
#                          gamma_n=GAMMA_N,
#                          n_step_return=N_STEP_RETURN)
#
# NUM_STATE = env_test.env.observation_space.shape[0]
# LOOPBACK_WINDOW = 4
# NUM_ACTIONS = env_test.env.action_space.n
# NONE_STATE = np.zeros(NUM_STATE)

# brain = Brain(LOOPBACK_WINDOW, NUM_STATE, NUM_ACTIONS, LOSS_V, LOSS_ENTROPY, MIN_BATCH, GAMMA_N, LEARNING_RATE, NONE_STATE)	# brain is global in A3C

# envs = [Environment() for i in range(THREADS)]
# opts = [Optimizer() for i in range(OPTIMIZERS)]

# for o in opts:
# 	o.start()

# for e in envs:
# 	e.start()

# time.sleep(RUN_TIME)

# for e in envs:
# 	e.stop()
# for e in envs:
# 	e.join()

# for o in opts:
# 	o.stop()
# for o in opts:
# 	o.join()

# print("Training finished")
# env_test.run()

