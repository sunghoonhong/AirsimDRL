'''
Author: Sunghoon Hong
Title:  randomly.py
Description:
    Random Agent for Airsim environment
'''
import time
import csv
import math
import argparse
import numpy as np
from airsim_env import Env

class RandomAgentDiscrete(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def get_action(self):
        action = np.random.choice(self.action_size)
        return action


class RandomAgentContinuous(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def get_action(self):
        action = np.random.uniform(-2, 2, self.action_size)
        return action


def interpret_action(action):
    scaling_factor = 0.5
    if action == 0:
        quad_offset = (0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, 0, scaling_factor)
    elif action == 4:
        quad_offset = (-scaling_factor, 0, 0)    
    elif action == 5:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 6:
        quad_offset = (0, 0, -scaling_factor)
    
    return quad_offset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--continuous', action='store_true')
    args = parser.parse_args()

    if args.continuous:
        agent = RandomAgentContinuous(3)
    else:
        agent = RandomAgentDiscrete(7)
    env = Env()

    episode = 0
    while True:
        done = False
        timestep = 0
        score = 0
        _ = env.reset()
        
        while not done:
            timestep += 1
            action = agent.get_action()
            if not args.continuous:
                action = interpret_action(action)
            _, reward, done, info = env.step(action)
            score += reward

            # stack history here
            if args.verbose:
                print('Step %d Action %s Reward %.2f Info %s:' % (timestep, action, reward, info))
        # done
        print('Ep %d: Step %d Score %.3f' % (episode, timestep, score))
        episode += 1

