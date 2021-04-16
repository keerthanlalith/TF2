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
model = models.load_model('MountainCar-v0-dqn.h5')


num_steps = 2500
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
        action = np.argmax(model.predict(np.array([obs]))[0])
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

    if episode_rew >-150:
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
MODE=1
# save state, action, nstate
if MODE == 0:
    filename = 'Data/TState.npy'
    pickle.dump(s, open(filename, 'wb'))
    filename = 'Data/TAction.npy'
    pickle.dump(a, open(filename, 'wb'))
    filename = 'Data/TNState.npy'
    pickle.dump(ns, open(filename, 'wb'))

else: 
    # save state, action, nstate
    filename = 'Data/State.npy'
    pickle.dump(s, open(filename, 'wb'))
    filename = 'Data/Action.npy'
    pickle.dump(a, open(filename, 'wb'))
    filename = 'Data/NState.npy'
    pickle.dump(ns, open(filename, 'wb'))



