import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD , Adam, RMSprop
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from tensorflow.keras.layers import LeakyReLU 
from matplotlib import pyplot
import matplotlib.pyplot as plt

import pickle 
import copy
import os
import gym
from feedback import *
import time
from collections import deque
import random
from rl_dqn import DeepQNetwork
from kinematic_model import Kinematic_Model


ENV = "CartPole-v1"
env = gym.make(ENV)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

########################################################
# RL Agent for Oracle
weights_file = 'rldqn_cartpole.h5'
oracle = DeepQNetwork(state_dim, action_dim, 0.001, 0.95, 1, 0.001, 0.995 )
oracle.load_weights(weights_file)
'''
TEST_Episodes=10
rewards = [] #Store rewards for graphing
#Test the agent that was trained
#   In this section we ALWAYS use exploit don't train any more
for e_test in range(TEST_Episodes):
    state,done = env.reset(), False
    state = np.reshape(state, [1, state_dim])
    tot_rewards = 0
    while not done:
        env.render()
        action = oracle.test_action(state)
        KM_nstate = Kinematic_Model(state[0],action)
        nstate, reward, done, _ = env.step(action)
        nstate = np.reshape( nstate, [1, state_dim])
        print("KM_NS : {}\t \t true_NS: {} Diff {}".format(KM_nstate, nstate[0],KM_nstate-nstate[0]))

        tot_rewards += reward
        #DON'T STORE ANYTHING DURING TESTING
        state = nstate
        #done: CartPole fell. 
        #t_test == 209: CartPole stayed upright
    rewards.append(tot_rewards)
    print("episode: {}/{}, score: {}, e: {}".format(e_test, TEST_Episodes, tot_rewards, 0))

env.close()

'''


########################################################
pause = 0 
while pause == 1:
    pause = 1

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

pause = 0 
while pause == 1:
    pause = 1

p_state = np.zeros((1,state_dim))
# Action for FDM to sample from
left = np.zeros((1,action_dim))
left[0][0] = 1
right = np.zeros((1,action_dim))
right[0][1] = 1

verbose = False
impove_FDM = True
#Buffer for FMD online improvement
FDM_buff_s =[]
FDM_buff_a =[]
FDM_buff_ns =[]
FDM_loss = 1000.0

AE_buff_s = []
AE_buff_ns = []
AE_loss = 1000.0

feedback_dict = {
      H_NULL: 0,
      H_UP: 0,
      H_DOWN: 0,
      H_LEFT: 1,
      H_RIGHT: 1,
      H_HOLD: 0,
      DO_NOTHING: 0
    }


# set which game to play
env = gym.make('CartPole-v0')
env.reset()
env.render()  # Make the environment visible

# Initialise Human feedback (call render before this)
human_feedback = Feedback(env)

num_steps = 5000
steps = 0
Episode = 1
total_reward = []

state = env.reset()
#state = np.reshape(state, [-1, state_dim])

prev_s = state
a = np.random.uniform(-1,1,action_dim)

print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)



while steps < num_steps:
    obs, terminal = env.reset(), False
    prev_state = obs
    print("Episode# ",Episode)
    Episode +=1
    episode_rew = 0

    # Iterate over the episode
    while((not terminal) and (not human_feedback.ask_for_done()) ):
        
        env.render()  # Make the environment visible
        time.sleep(0.1)
        # Get feedback signal
        h_fb = human_feedback.get_h() 
        
        
        if (feedback_dict.get(h_fb) != 0):  # if feedback is not zero i.e. is valid
            # Update policy
            #oracle_action = oracle.test_action(np.reshape(obs, [1, state_dim]))
            #h_fb = oracle_action + 3
            #print("Feedback", h_fb)

            # Get new state transition label using feedback
            state_corrected = copy.deepcopy(obs)
            if (h_fb == H_LEFT): # PUSH CART TO LEFT
                print("Move left")
                state_corrected = Kinematic_Model(state,0)
                #state_corrected[0] -= 0.01 # correction in pos
                #state_corrected[1] -= 0.2 # correction in vel
                #state_corrected[2] += 0.01 # correction in angle
                #state_corrected[3] += 0.27 # correction in anglar vel
            elif (h_fb == H_RIGHT):# PUSH CART TO RIGHT
                print("Move right")
                state_corrected = Kinematic_Model(state,1)
                #state_corrected[0] += 0.01 # correction in pos
                #state_corrected[1] += 0.2 # correction in vel
                #state_corrected[2] -= 0.01 # correction in angle
                #state_corrected[3] -= 0.27 # correction in anglar vel
            
            # Add state transition pair to demo buffer
            AE_buff_s.append(obs)
            AE_buff_ns.append(state_corrected)

            print("State           ",obs)            
            print("state_corrected ",state_corrected)
            pred_ns = np.reshape(state_corrected, [-1, state_dim])

            '''
            # Update policy (immediate)
            temp_s = np.array(p_state)
            temp_ns = np.array(state_corrected)
            history_AE = AE.fit(x=temp_s, y=temp_ns, verbose=1)    


            # Train with batch from Demo buffer (if enough entries exist)
            num = len(AE_buff_s)
            print("num",num)
            if(num >= 64): # batch size 64
                print("Training AE")
                temp_s = np.array(AE_buff_s)
                temp_ns = np.array(AE_buff_ns)
                # do you need to run for 5 epochs
                history_AE = AE.fit(x=temp_s, y=temp_ns,batch_size=64,shuffle=True, verbose=1)
            '''

        else:
            # Use current policy
            print("Using current policy")
            p_state = np.reshape(obs, [-1, state_dim])
            pred_ns = AE.predict(p_state)

        # Get action from ifdm
        FDM_ns_l = np.squeeze(FDM.predict([p_state,left]))
        FDM_ns_r = np.squeeze(FDM.predict([p_state,right]))
        FDM_ns_both = np.array([FDM_ns_l,FDM_ns_r])
        state_diff  = np.abs(FDM_ns_both-pred_ns)
        cost = np.sum(state_diff,axis=1)
        action_from_IDM = np.argmin(cost, axis=0)
        action = action_from_IDM

        prev_state = obs
        obs, reward, terminal, _ = env.step(action)
        episode_rew += reward

        # Add state transition pair to demo buffer
        FDM_buff_s.append(prev_state)
        FDM_buff_a.append(action)
        FDM_buff_ns.append(obs)

        if verbose ==True:
            print("Curr state                       ",prev_state)
            print("True next state                  ",obs)
            print("AE pred Nstate                   ",pred_ns)
            print("FDM both next state              ",FDM_ns_both)
            print("cost                             ",cost)
            print("partial cost                     ",par_cost)
            print("action from IDM  full cost       ",action_from_IDM)
        if action == 0:
            print("FDM left")        #0 Push cart to the left
        else:
            print("FDM right")        #1 Push cart to the right
            print("")
            
        steps += 1
    
    total_reward.append(episode_rew)
    #print('Episode #%d Reward %d' % (Episode, episode_rew))
    print("## episode: {}, Reward: {}".format(Episode, episode_rew))

    #Train Next State predictor
    # Train with batch from Demo buffer (if enough entries exist)
    num = len(AE_buff_s)
    if(num >= 64) and AE_loss > 0.0001: # batch size 64
        print("Training AE")
        # do you need to run for 5 epochs
        history_AE = AE.fit(x=np.array(AE_buff_s), y=np.array(AE_buff_ns),batch_size=64,epochs=5,shuffle=True, verbose=False)
        AE_loss = history_AE.history['loss'][-1]
        print("AE loss",AE_loss)


    if FDM_loss > 0.0001:
        print("Training FDM")
        #Train FDM every episode,
        temp_s = np.array(FDM_buff_s)
        temp_a = np.zeros((len(FDM_buff_a),action_dim))
        for i in range(len(FDM_buff_a)):
            temp_a[i][FDM_buff_a[i]] =1
        history_FDM=FDM.fit(x=[temp_s,temp_a], y=np.array(FDM_buff_ns),epochs=5,batch_size=32, shuffle=True,verbose=False)
        FDM_loss = history_FDM.history['loss'][-1]
        print("FDM loss",FDM_loss)

for i in range(Episode):
    print("episode: {}, Reward: {}".format(i, total_reward[i-1]))

total_reward =np.array(total_reward)
rolling_average = np.convolve(total_reward, np.ones(100)/100)

plt.plot(total_reward)
plt.plot(rolling_average, color='black')
plt.axhline(y=195, color='r', linestyle='-') #Solved Line
plt.xlim( (0,Episode) )
plt.ylim( (0,220) )
plt.show()
