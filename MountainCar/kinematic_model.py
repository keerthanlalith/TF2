import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np




def Kinematic_Model(state, action):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5
    
    force=0.001
    gravity=0.0025

    position, velocity = state
    velocity += (action-1)*force + math.cos(3*position)*(-gravity)
    velocity = np.clip(velocity, -max_speed, max_speed)
    position += velocity
    position = np.clip(position, min_position, max_position)
    if (position== min_position and velocity<0): velocity = 0

    state = (position, velocity)


    return np.array(state)

