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

trial_no = '9a7'
if not os.path.exists('log_files/'+trial_no):
    os.makedirs('log_files/'+trial_no)
ENV = "CartPole-v0"
env = gym.make(ENV)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

########################################################
# RL Agent for Oracle
weights_file = 'rldqn_cartpole.h5'
oracle = DeepQNetwork(state_dim, action_dim, 0.001, 0.95, 1, 0.001, 0.995 )
oracle.load_weights(weights_file)

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
#############################################################################################


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

num_steps = 3000
steps = 0
Episode = 1
total_reward = []
feedback_rate = []

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



while Episode < 100:
    obs, terminal = env.reset(), False
    prev_state = obs
    print("Episode# ",Episode)
    
    episode_rew = 0
    h_counter = 0
    t_counter = 0

    # Iterate over the episode
    while((not terminal) and (not human_feedback.ask_for_done()) ):
        
        env.render()  # Make the environment visible
        time.sleep(0.05)
        
        # Get feedback signal
        h_fb = human_feedback.get_h()
        if  h_fb == 1:
            print("Oracle")
            oracle_action = oracle.test_action(np.reshape(obs, [1, state_dim]))
            h_fb = oracle_action + 3
        
        
        if (feedback_dict.get(h_fb) != 0):  # if feedback is not zero i.e. is valid
            # Update policy
            oracle_action = oracle.test_action(np.reshape(obs, [1, state_dim]))
            print("oracle_action ",oracle_action)
            #h_fb = oracle_action + 3
            #print("Feedback", h_fb)

            h_counter += 1

            # Get new state transition label using feedback
            state_corrected = copy.deepcopy(obs)
            if (h_fb == H_LEFT): # PUSH CART TO LEFT
                print("Feedback left\t",end =" ")
                #state_corrected = Kinematic_Model(state,0)
                state_corrected[0] -= 0.1 # correction in pos
                #state_corrected[1] -= 0.2 # correction in vel
                #state_corrected[2] += 0.01 # correction in angle
                #state_corrected[3] += 0.27 # correction in anglar vel
            elif (h_fb == H_RIGHT):# PUSH CART TO RIGHT
                print("Feedback right\t",end =" ")
                #state_corrected = Kinematic_Model(state,1)
                state_corrected[0] += 0.1 # correction in pos
                #state_corrected[1] += 0.2 # correction in vel
                #state_corrected[2] -= 0.01 # correction in angle
                #state_corrected[3] -= 0.27 # correction in anglar vel
            
            # Add state transition pair to demo buffer
            AE_buff_s.append(obs)
            AE_buff_ns.append(state_corrected)
            print()
            #print("State           ",obs)            
            #print("state_corrected ",state_corrected)
            pred_ns = np.reshape(state_corrected, [-1, state_dim])

            if(len(AE_buff_s) >= 64): # batch size 64
                print("Training AE")
                # do you need to run for 5 epochs
                history_AE = AE.fit(x=np.array(AE_buff_s), y=np.array(AE_buff_ns),batch_size=64,shuffle=True, verbose=False)
                AE_loss = history_AE.history['loss'][-1]
                print("AE loss",AE_loss)

        else:
            # Use current policy
            print("Policy \t\t",end =" ")
            p_state = np.reshape(obs, [-1, state_dim])
            pred_ns = AE.predict(p_state)

        # Get action from ifdm
        FDM_ns_l = np.squeeze(FDM.predict([p_state,left]))
        FDM_ns_r = np.squeeze(FDM.predict([p_state,right]))
        FDM_ns_l = Kinematic_Model(obs,0)
        FDM_ns_r = Kinematic_Model(obs,1)
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
        verbose =True
        if verbose ==True:
            print("Curr state                       ",prev_state)
            print("True next state                  ",obs)
            print("AE pred Nstate                   ",pred_ns)
            print("FDM both next state              ",FDM_ns_both)
            print("cost                             ",cost)
            print("action from IDM  full cost       ",action_from_IDM)

        if action == 0:
            print("FDM left")        #0 Push cart to the left
        else:
            print("FDM right")        #1 Push cart to the right
        print("")
            
        steps += 1
        t_counter+=1

    
    
    feedback_rate.append(h_counter/t_counter)
    total_reward.append(episode_rew)

    #print('Episode #%d Reward %d' % (Episode, episode_rew))
    print("## episode: {}, Reward: {}, h_counter: {}, t_counter: {}, feedback_rate: {} ".format(Episode, episode_rew,h_counter,t_counter,h_counter/t_counter))
    Episode +=1

    #Train Next State predictor
    # Train with batch from Demo buffer (if enough entries exist)
    num = len(AE_buff_s)
    if(num >= 64) and AE_loss > 0.0001: # batch size 64
        print("Training AE")
        # do you need to run for 5 epochs
        history_AE = AE.fit(x=np.array(AE_buff_s), y=np.array(AE_buff_ns),batch_size=64,epochs=10,shuffle=True, verbose=False)
        AE_loss = history_AE.history['loss'][-1]
        print("AE loss",AE_loss)


    if FDM_loss > 0.0001:
        print("Training FDM")
        #Train FDM every episode,
        temp_s = np.array(FDM_buff_s)
        temp_a = np.zeros((len(FDM_buff_a),action_dim))
        for i in range(len(FDM_buff_a)):
            temp_a[i][FDM_buff_a[i]] =1
        history_FDM=FDM.fit(x=[temp_s,temp_a], y=np.array(FDM_buff_ns),epochs=10,batch_size=32, shuffle=True,verbose=False)
        FDM_loss = history_FDM.history['loss'][-1]
        print("FDM loss",FDM_loss)

for i in range(Episode):
    print("episode: {}, Reward: {}".format(i, total_reward[i-1]))

total_reward =np.array(total_reward)
rolling_average = np.convolve(total_reward, np.ones(100)/100)
feedback_rate = np.array(feedback_rate) * 100

plt.plot(total_reward)
plt.plot(rolling_average, color='black')
plt.plot(feedback_rate, color='green')
plt.axhline(y=195, color='r', linestyle='-') #Solved Line
plt.xlim( (0,Episode) )
plt.ylim( (0,220) )
plt.show()


print("Saving Reward")
filename = 'log_files/'+trial_no+'/total_reward.npy'
pickle.dump(total_reward, open(filename, 'wb'))
filename = 'log_files/'+trial_no+'/feedback_rate.npy'
pickle.dump(feedback_rate, open(filename, 'wb'))

AE.save('log_files/'+trial_no+'/AE')
FDM.save('log_files/'+trial_no+'/FDM')

'''

else: 
    # save state, action, nstate
    filename = 'Data/State.npy'
    pickle.dump(s, open(filename, 'wb'))
    filename = 'Data/Action.npy'
    pickle.dump(a, open(filename, 'wb'))
    filename = 'Data/NState.npy'
    pickle.dump(ns, open(filename, 'wb'))
    filename = 'Data/Diff.npy'
    pickle.dump(d,open(filename,'wb'))
'''