import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np




def Kinematic_Model(state, action):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates
    kinematics_integrator = 'euler'
    x, x_dot, theta, theta_dot = state
    force = force_mag if action==1 else -force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
    xacc  = temp - polemass_length * thetaacc * costheta / total_mass
    if kinematics_integrator == 'euler':
        x  = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
    else: # semi-implicit euler
        x_dot = x_dot + tau * xacc
        x  = x + tau * x_dot
        theta_dot = theta_dot + tau * thetaacc
        theta = theta + tau * theta_dot
    state = (x,x_dot,theta,theta_dot)


    return np.array(state)

