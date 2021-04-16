# test_dqn.py
# https://geektutu.com
import time
import gym
import numpy as np
from tensorflow.keras import models
import pickle
import time

s = []
a = []
ns =[]
env = gym.make('MountainCar-v0')

num_steps = 1000
steps = 0
episode =0
while steps < num_steps:
    print("episode #",episode)
    s_e = []
    a_e = []
    ns_e = []
    episode_len = 0
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        s_e.append(obs)
        time.sleep(0.01)
        action = env.action_space.sample()
        if action == 0:
            print("left")
        elif action== 1:
            print("stop")
        else:
            print("right")

        obs, rew, done, _ = env.step(action)
        episode_rew += rew
        a_e.append(action)
        ns_e.append(obs)
        episode_len = episode_len + 1


    
    print('episode_rew:', episode_rew)
    print('episode_len:', episode_len)
    time.sleep(1)

    s.extend(s_e)
    a.extend(a_e)
    ns.extend(ns_e)
    print("Saving data")
    steps = steps + episode_len
        
    episode += 1

env.close()
print("End all")
s=np.array(s)
a=np.array(a)
ns=np.array(ns)
filename = 'Data/RState.npy'
pickle.dump(s, open(filename, 'wb'))
filename = 'Data/RAction.npy'
pickle.dump(a, open(filename, 'wb'))
filename = 'Data/RNState.npy'
pickle.dump(ns, open(filename, 'wb'))



