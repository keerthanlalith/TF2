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


# Import Data
trial_no = '9a1'
filename = 'log_files/'+trial_no+'/feedback_rate.npy'
trial1 = pickle.load(open(filename, 'rb'))

trial_no = '9a2'
filename = 'log_files/'+trial_no+'/feedback_rate.npy'
trial2 = pickle.load(open(filename, 'rb'))

trial_no = '9a3'
filename = 'log_files/'+trial_no+'/feedback_rate.npy'
trial3 = pickle.load(open(filename, 'rb'))

trial_no = '9a4'
filename = 'log_files/'+trial_no+'/feedback_rate.npy'
trial4 = pickle.load(open(filename, 'rb'))

trial_no = '9a5'
filename = 'log_files/'+trial_no+'/feedback_rate.npy'
trial5 = pickle.load(open(filename, 'rb'))

import copy
avg = copy.deepcopy(trial1)

for i in range (len(trial1)):
    all=[trial1[i],trial2[i],trial3[i]]
    avg[i] = np.average(all)

Episode = np.arange(100)
plt.plot(trial1)
plt.plot(trial2)
plt.plot(trial3)
plt.plot(trial4)
plt.plot(trial5)
plt.plot(avg,'k')

plt.axhline(y=195, color='r', linestyle='-') #Solved Line
plt.xlim( (0,100) )
plt.ylim( (0,220) )
plt.show()

