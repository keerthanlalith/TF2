import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD , Adam, RMSprop
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from tensorflow.keras.layers import LeakyReLU 
from matplotlib import pyplot
import pickle 
import copy
import os
import gym


state_dim = 4
action_dim = 2


# Import Data
filename = 'Data/NState.npy'
ns = pickle.load(open(filename, 'rb'))
filename = 'Data/Action.npy'
a = pickle.load(open(filename, 'rb'))
filename = 'Data/State.npy'
s = pickle.load(open(filename, 'rb'))
filename = 'Data/Diff.npy'
d = pickle.load(open(filename, 'rb'))

temp = np.zeros((a.size,2))
for i in range(len(a)):
    temp[i][a[i]] = 1
a = temp

# test set
filename = 'Data/TNState.npy'
test_ns = pickle.load(open(filename, 'rb'))
filename = 'Data/TAction.npy'
test_a = pickle.load(open(filename, 'rb'))
filename = 'Data/TState.npy'
test_s = pickle.load(open(filename, 'rb'))
filename = 'Data/TDiff.npy'
test_d = pickle.load(open(filename, 'rb'))

temp = np.zeros((test_a.size,2))
for i in range(len(test_a)):
    temp[i][test_a[i]] = 1
test_a = temp


if not os.path.exists("saved_model"):
    os.makedirs("saved_model")

reuse =True
if reuse ==True:
    print("Using pretrained models")
    AE = keras.models.load_model('saved_model/AE')
    FDM = keras.models.load_model('saved_model/FDM')
else:
    # training
    print("Training started")
    history_AE = AE.fit(x=s, y=ns,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    verbose=1,
                    validation_data=(test_s, test_ns))

    print("AE Training Done")
    history_FDM = FDM.fit(x=[s,a], y=ns,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    verbose=1,
                    validation_data=([test_s,test_a], test_ns))

    print("FDM Training Done")
    history_dict_AE = history_AE.history
    history_dict_FDM = history_FDM.history
    AE.save('saved_model/AE')
    FDM.save('saved_model/FDM')

print("Eval")
# evaluate the model
_, train_mse = AE.evaluate(s, ns, verbose=0)
_, test_mse = AE.evaluate(test_s, test_ns, verbose=0)
print('Train: %.6f, Test: %.6f' % (train_mse, test_mse))


_, train_mse = FDM.evaluate([s,a], ns, verbose=0)
_, test_mse = FDM.evaluate([test_s,test_a], test_ns, verbose=0)
print('Train: %.6f, Test: %.6f' % (train_mse, test_mse))

pause = 0 
while pause == 1:
    pause = 1

p_state = np.zeros((1,state_dim))
left = np.zeros((1,action_dim))
left[0][0] = 1
right = np.zeros((1,action_dim))
right[0][1] = 1


ENV = "CartPole-v0"
num_steps = 300
verbose =True
Episode = 0

env = gym.make(ENV)
steps = 0
total_reward = 0
episodes = 0
while steps < num_steps:
    obs, done = env.reset(), False
    episode_rew = 0
    prev_state = obs
    print("Episode# ",Episode)
    while not done:
        env.render()
        for i in range(4):
            p_state[0][i] = obs[i]
        pred_ns = AE.predict(p_state)    
        FDM_ns_l = np.squeeze(FDM.predict([p_state,left]))
        FDM_ns_r = np.squeeze(FDM.predict([p_state,right]))
        FDM_ns_both = np.array([FDM_ns_l,FDM_ns_r])
        state_diff  = np.abs(FDM_ns_both-pred_ns)
        par_cost =np.array([state_diff[0][0]+state_diff[0][2],state_diff[1][0]+state_diff[1][2]])
        cost = np.sum(state_diff,axis=1)
        action_from_IDM = np.argmin(cost, axis=0)
        action_from_pIDM = np.argmin(par_cost, axis=0)
        action = action_from_pIDM
        #0 Push cart to the left
        #1 Push cart to the right
        #print(action)
        prev_state = obs
        obs, rew, done, _ = env.step(action)

        if verbose ==True:
            print("Curr state                       ",prev_state)
            print("True next state                  ",obs)
            print("AE pred Nstate                   ",pred_ns)
            print("FDM both next state              ",FDM_ns_both)
            print("cost                             ",cost)
            print("partial cost                     ",par_cost)
            print("action from IDM  full cost       ",action_from_IDM)
            print("action from IDM  partial cost    ",action_from_pIDM)
            if action == 0:
                print("left")
            else:
                print("right")
            print("")

        episode_rew += rew
        steps += 1
    #print(steps)
    Episode +=1
    print("Episode reward", episode_rew)
    total_reward += episode_rew
    episodes += 1.
print("Average reward", total_reward / episodes)
