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
ENV ='MountainCar_rev-v0'
env = gym.make(ENV)

env.reset()
for _ in range(1000):
    env.render()
    action = 2
    #if np.random.uniform(0,1)<=0.5:
    #    action = 2
    if action == 0:
        print("Left")
    else :
        print("right")

    env.step(action) # take a random action
env.close()