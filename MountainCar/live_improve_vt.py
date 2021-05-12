import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD , Adam, RMSprop
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from tensorflow.keras.layers import LeakyReLU 
from matplotlib import pyplot
import matplotlib.pyplot as plt

from mountaincar_soln import Agent

import pickle 
import copy
import os
import gym
from feedback import *
import time
from collections import deque
import random
import getch

#from rl_dqn import DeepQNetwork
from kinematic_model import Kinematic_Model

trial_no = '4h1'
if not os.path.exists('Log_files/'+trial_no):
    os.makedirs('Log_files/'+trial_no)
ENV = "MountainCar-v0"
env = gym.make(ENV)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

########################################################
AE = keras.models.load_model('Log_files/'+trial_no+'/AE')
FDM =keras.models.load_model('Log_files/'+trial_no+'/FDM')




'''
# This model maps an input to its next state
# Input state
AE_state = keras.Input(shape=(state_dim,),name="AE_state")
# 2layer neural network to predict the next state
encoded = Dense(32,name="dense1_NS")(AE_state)
encoded = LeakyReLU(alpha=0.2,name="LeakyRelu1_NS")(encoded)
encoded = Dense(32,name="dense2_NS")(encoded)
encoded = LeakyReLU(alpha=0.2,name="LeakyRelu2_NS")(encoded)
n_state = layers.Dense(state_dim,name="dense3_NS")(encoded)
AE = keras.Model(inputs=AE_state, outputs=n_state,name="AE")

#print(AE.summary())
#tf.keras.utils.plot_model(AE, to_file='AE_model_plot.png', show_shapes=True, show_layer_names=True)

opt_AE = tf.keras.optimizers.RMSprop(learning_rate=0.00015)
AE.compile(loss='mean_squared_error', optimizer=opt_AE, metrics=['mse'])





# This model maps an input & action to its next state
# Input state
curr_state = keras.Input(shape=(state_dim,),name="curr_state")
curr_action = keras.Input(shape=(action_dim,),name="curr_action")
# FDM model
curr_state_action = concatenate([curr_state, curr_action])
fdm_h1 = Dense(16,name="dense1_FDM")(curr_state_action)
fdm_h1 = LeakyReLU(alpha=0.2,name="LeakyRelu1_FDM")(fdm_h1)
fdm_h2 = Dense(16,name="dense2_FDM")(fdm_h1)
fdm_h2 = LeakyReLU(alpha=0.2,name="LeakyRelu2_FDM")(fdm_h2)
fdm_pred_state = layers.Dense(state_dim,name="dense3_FDM")(fdm_h2)
FDM = keras.Model(inputs=[curr_state,curr_action], outputs=fdm_pred_state,name="FDM")

#print(FDM.summary())
#tf.keras.utils.plot_model(FDM, to_file='FDM_model_plot.png', show_shapes=True, show_layer_names=True)

opt_FDM = tf.keras.optimizers.RMSprop(learning_rate=0.00015)
FDM.compile(loss='mean_squared_error', optimizer=opt_FDM, metrics=['mse'])
################################################################################################
'''

p_state = np.zeros((1,state_dim))
# Action for FDM to sample from
left = np.zeros((1,action_dim))
left[0][0] = 1
right = np.zeros((1,action_dim))
right[0][2] = 1

def Test_policy():
    print("[--------------Test Policy--------------]")

    Test_episode = 1
    while Test_episode  <= 10:

        obs, terminal = env.reset(), False
        prev_state = obs
        episode_rew = 0

        while not terminal :
            env.render()  # Make the environment visible
            time.sleep(0.01)
            
            p_state = np.reshape(obs, [-1, state_dim])
            pred_ns = AE.predict(p_state)
            FDM_ns_l = np.squeeze(FDM.predict([p_state,left]))
            FDM_ns_r = np.squeeze(FDM.predict([p_state,right]))
            FDM_ns_both = np.array([FDM_ns_l,FDM_ns_r])
            state_diff  = np.abs(FDM_ns_both-pred_ns)
            cost = np.sum(state_diff,axis=1)
            action_from_IDM = np.argmin(cost, axis=0)*2
            action = action_from_IDM

            obs, reward, terminal, _ = env.step(action)
            episode_rew += reward

        print("Background Trial`# {} Reward : {}".format(Test_episode,episode_rew))
        Test_episode+=1
    print("[---------------------------------------]")


Test_policy()
print("Done!!")
