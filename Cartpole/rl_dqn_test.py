#Imports
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import random
from rl_dqn import DeepQNetwork,learning_rate,discount_rate

#Create Gym
from gym import wrappers
env = gym.make('CartPole-v1')
env.seed(50) #Set the seed to keep the environment consistent across runs

weights_file = 'rldqn_cartpole.h5'

state_dim = env.observation_space.shape[0] #This is only 4
action_dim = env.action_space.n #Actions
print("Deifning dqn")
dqn1 = DeepQNetwork(state_dim, action_dim, 0.001, 0.95, 1, 0.001, 0.995 )
print("loadin dqn")
dqn1.load_weights(weights_file)

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
        action = dqn1.test_action(state)
        nstate, reward, done, _ = env.step(action)
        nstate = np.reshape( nstate, [1, state_dim])
        tot_rewards += reward
        #DON'T STORE ANYTHING DURING TESTING
        state = nstate
        #done: CartPole fell. 
        #t_test == 209: CartPole stayed upright
    rewards.append(tot_rewards)
    print("episode: {}/{}, score: {}, e: {}".format(e_test, TEST_Episodes, tot_rewards, 0))

env.close()

