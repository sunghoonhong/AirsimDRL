'''
Author: Sunghoon Hong
Title: ra2c.py
Description:
    Recurrent Advantage Actor Critic Agent for Airsim
Detail:
    - not use join()
    - reset for zero-image error
    - tensorflow v1 + keras
    - hard update for target critic

'''


import os
import csv
import time
import random
import argparse
from copy import deepcopy
from datetime import datetime as dt
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import TimeDistributed, BatchNormalization, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU
from keras.optimizers import Adam
from keras.models import Model
from PIL import Image

from airsim_env import Env, ACTION

agent_name = 'ra2c'


class A2CAgent(object):
    
    def __init__(self, state_size, action_size, actor_lr, critic_lr, tau,
                gamma, lambd, entropy, horizon, load_model):
        self.state_size = state_size
        self.action_size = action_size
        self.vel_size = 3
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.lambd = lambd
        self.entropy = entropy
        self.horizon = horizon

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.actor, self.critic = self.build_model()
        _, self.target_critic = self.build_model()
        self.actor_update = self.build_actor_optimizer()
        self.critic_update = self.build_critic_optimizer()
        self.sess.run(tf.global_variables_initializer())
        if load_model:
            self.load_model('./save_model/'+ agent_name)
        
        self.target_critic.set_weights(self.critic.get_weights())

        self.states, self.actions, self.rewards = [], [], []

    def build_model(self):
        # shared network
        image = Input(shape=self.state_size)
        image_process = BatchNormalization()(image)
        image_process = TimeDistributed(Conv2D(32, (8, 8), activation='elu', padding='same', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(32, (5, 5), activation='elu', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(8, (1, 1), activation='elu', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(Flatten())(image_process)
        image_process = GRU(64, kernel_initializer='he_normal', use_bias=False)(image_process)
        image_process = BatchNormalization()(image_process)
        image_process = Activation('tanh')(image_process)

        # vel process
        vel = Input(shape=[self.vel_size])
        # vel_process = Dense(6, kernel_initializer='he_normal', use_bias=False)(vel)
        # vel_process = BatchNormalization()(vel_process)
        # vel_process = Activation('tanh')(vel_process)
        
        state_process = image_process

        # Actor
        policy = Dense(128, kernel_initializer='he_normal', use_bias=False)(state_process)
        policy = ELU()(policy)
        policy = BatchNormalization()(policy)
        policy = Dense(self.action_size, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-2e-3, maxval=2e-3))(policy)
        actor = Model(inputs=[image, vel], outputs=policy)

        # Critic
        value = Dense(128, kernel_initializer='he_normal', use_bias=False)(state_process)
        value = ELU()(value)
        value = BatchNormalization()(value)
        value = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(value)
        critic = Model(inputs=[image, vel], outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()
        
        return actor, critic

    def build_actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-6) * advantages
        cross_entropy = -K.mean(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-6), axis=1)
        entropy = K.mean(entropy)

        loss = cross_entropy + self.entropy * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input[0], self.actor.input[1], action, advantages],
                           [loss], updates=updates)
        return train

    def build_critic_optimizer(self):
        y = K.placeholder(shape=(None, 1))

        value = self.critic.output
        # MSE loss
        loss = K.mean(K.square(y - value))
        # # Huber loss
        # error = K.abs(y - value)
        # quadratic = K.clip(error, 0.0, 1.0)
        # linear = error - quadratic
        # loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input[0], self.critic.input[1], y],
                           [loss], updates=updates)
        return train

    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        policy = np.array(policy)
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action, policy

    def train_model(self, next_state, done):
        images = np.zeros([len(self.states) + 1] + self.state_size, dtype=np.float32)
        vels = np.zeros([len(self.states) + 1, self.vel_size], dtype=np.float32)
        for i in range(len(self.states)):
            images[i], vels[i] = self.states[i]
        images[-1], vels[-1] = next_state
        states = [images, vels]
        values = self.target_critic.predict(states)
        values = np.reshape(values, len(values))

        advantage = np.zeros_like(self.rewards, dtype=np.float32)

        gae = 0
        if done:
            values[-1] = np.float32([0])
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] - values[t]
            gae = delta + self.gamma * self.lambd * gae
            advantage[t] = gae

        target_val = advantage + values[:-1]
        target_val = target_val.reshape((-1, 1))
        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-6)

        states = [images[:-1], vels[:-1]]
        actor_loss = self.actor_update(states + [self.actions, advantage])
        critic_loss = self.critic_update(states + [target_val])
        self.clear_sample()
        return actor_loss[0], critic_loss[0]
    
    def append_sample(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def clear_sample(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def update_target_model(self):
        self.target_critic.set_weights(self.critic.get_weights())
        
    def load_model(self, name):
        if os.path.exists(name + '_actor.h5'):
            self.actor.load_weights(name + '_actor.h5')
            print('Actor loaded')
        if os.path.exists(name + '_critic.h5'):
            self.critic.load_weights(name + '_critic.h5')
            print('Critic loaded')

    def save_model(self, name):
        self.actor.save_weights(name + '_actor.h5')
        self.critic.save_weights(name + '_critic.h5')

'''
Environment interaction
'''

def transform_input(responses, img_height, img_width):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = np.array(np.clip(255 * 3 * img1d, 0, 255), dtype=np.uint8)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    image = Image.fromarray(img2d)
    image = np.array(image.resize((img_width, img_height)).convert('L'))
    cv2.imwrite('view.png', image)
    image = np.float32(image.reshape(1, img_height, img_width, 1))
    image /= 255.0
    return image

def interpret_action(action):
    scaling_factor = 1.
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

    # CUDA config
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--play',       action='store_true')
    parser.add_argument('--img_height', type=int,   default=72)
    parser.add_argument('--img_width',  type=int,   default=128)
    parser.add_argument('--actor_lr',   type=float, default=5e-5)
    parser.add_argument('--critic_lr',  type=float, default=1e-4)
    parser.add_argument('--tau',        type=float, default=0.1)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--lambd',      type=float, default=0.90)
    parser.add_argument('--entropy',    type=float, default=1e-3)
    parser.add_argument('--horizon',    type=int,   default=32)
    parser.add_argument('--seqsize',    type=int,   default=5)
    parser.add_argument('--target_rate',type=int,   default=1000)

    args = parser.parse_args()

    if not os.path.exists('save_graph/'+ agent_name):
        os.makedirs('save_graph/'+ agent_name)
    if not os.path.exists('save_stat'):
        os.makedirs('save_stat')
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    # Make RL agent
    state_size = [args.seqsize, args.img_height, args.img_width, 1]
    action_size = 7
    agent = A2CAgent(
        state_size=state_size,
        action_size=action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        lambd=args.lambd,
        entropy=args.entropy,
        horizon=args.horizon,
        load_model=args.load_model
    )

    # Train
    episode = 0
    highscoreY = 0.
    if os.path.exists('save_stat/'+ agent_name + '_stat.csv'):
        with open('save_stat/'+ agent_name + '_stat.csv', 'r') as f:
            read = csv.reader(f)
            episode = int(float(next(reversed(list(read)))[0]))
            print('Last episode:', episode)
            episode += 1
    if os.path.exists('save_stat/'+ agent_name + '_highscore.scv'):
        with open('save_stat/'+ agent_name + '_highscore.csv', 'r') as f:
            read = csv.reader(f)
            highscoreY = float(next(reversed(list(read)))[0])
            print('Best Y:', highscoreY)
    stats = []

    env = Env()

    if args.play:
        while True:
            try:
                done = False
                bug = False

                # stats
                bestY, timestep, score, pmax = 0., 0, 0., 0.

                observe = env.reset()
                image, vel = observe
                try:
                    image = transform_input(image, args.img_height, args.img_width)
                except:
                    continue
                history = np.stack([image] * args.seqsize, axis=1)
                vel = vel.reshape(1, -1)
                state = [history, vel]
                while not done:
                    timestep += 1
                    # snapshot = np.zeros([0, args.img_width, 1])
                    # for snap in state[0][0]:
                    #     snapshot = np.append(snapshot, snap, axis=0)
                    # snapshot *= 128
                    # snapshot += 128
                    # cv2.imshow('%s' % timestep, np.uint8(snapshot))
                    # cv2.waitKey(0)
                    action, policy = agent.get_action(state)
                    real_action = interpret_action(action)
                    observe, reward, done, info = env.step(real_action)
                    image, vel = observe
                    try:
                        image = transform_input(image, args.img_height, args.img_width)
                    except:
                        bug = True
                        break
                    history = np.append(history[:, 1:], [image], axis=1)
                    vel = vel.reshape(1, -1)
                    next_state = [history, vel]

                    # stats
                    pmax += float(np.amax(policy))
                    score += reward
                    if info['Y'] > bestY:
                        bestY = info['Y']
                    print('%s' % (ACTION[action]), end='\r', flush=True)

                    if args.verbose:
                        print('Step %d Action %s Reward %.2f Info %s:' % (timestep, real_action, reward, info['status']))

                    state = next_state

                if bug:
                    continue
                
                pmax /= timestep

                # done
                print('Ep %d: BestY %.3f Step %d Score %.2f Pmax %.2f'
                        % (episode, bestY, timestep, score, pmax))

                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
    else:
        # Train
        time_limit = 600
        highscoreY = 0.
        if os.path.exists('save_stat/'+ agent_name + '_stat.csv'):
            with open('save_stat/'+ agent_name + '_stat.csv', 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
                print('Last episode:', episode)
                episode += 1
        if os.path.exists('save_stat/'+ agent_name + '_highscore.csv'):
            with open('save_stat/'+ agent_name + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscoreY = float(next(reversed(list(read)))[0])
                print('Best Y:', highscoreY)
        global_step = 0
        while True:
            try:
                done = False
                bug = False

                # stats
                bestY, timestep, score, pmax = 0., 0, 0., 0.
                t, actor_loss, critic_loss = 0, 0., 0.
                observe = env.reset()
                image, vel = observe
                try:
                    image = transform_input(image, args.img_height, args.img_width)
                except:
                    continue
                history = np.stack([image] * args.seqsize, axis=1)
                vel = vel.reshape(1, -1)
                state = [history, vel]
                while not done and timestep < time_limit:
                    t += 1
                    timestep += 1
                    global_step += 1
                    if global_step >= args.target_rate:
                        agent.update_target_model()
                        global_step = 0
                    action, policy = agent.get_action(state)
                    real_action = interpret_action(action)
                    observe, reward, done, info = env.step(real_action)
                    image, vel = observe
                    try:
                        if timestep < 3 and info['status'] == 'landed':
                            raise Exception
                        image = transform_input(image, args.img_height, args.img_width)
                    except:
                        bug = True
                        break
                    history = np.append(history[:, 1:], [image], axis=1)
                    vel = vel.reshape(1, -1)
                    next_state = [history, vel]
                    agent.append_sample(state, action, reward)

                    # stats
                    score += reward
                    pmax += float(np.amax(policy))
                    if info['Y'] > bestY:
                        bestY = info['Y']

                    print('%s | %.3f | %.3f' % (ACTION[action], policy[action], policy[2]), end='\r')

                    if args.verbose:
                        print('Step %d Action %s Reward %.2f Info %s:' % (timestep, action, reward, info['status']))

                    if t >= args.horizon or done:
                        t = 0
                        a_loss, c_loss = agent.train_model(next_state, done)
                        actor_loss += float(a_loss)
                        critic_loss += float(c_loss)

                    state = next_state

                if bug:
                    continue

                # done
                pmax /= timestep
                actor_loss /= (timestep // args.horizon + 1)
                critic_loss /= (timestep // args.horizon + 1)

                if args.verbose or episode % 10 == 0:
                    print('Ep %d: BestY %.3f Step %d Score %.2f Pmax %.2f'
                            % (episode, bestY, timestep, score, pmax))
                stats = [
                    episode, timestep, score, bestY, \
                    pmax, actor_loss, critic_loss, info['level'], info['status']
                ]

                # log stats
                with open('save_stat/'+ agent_name + '_stat.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['%.4f' % s if type(s) is float else s for s in stats])
                if highscoreY < bestY:
                    highscoreY = bestY
                    with open('save_stat/'+ agent_name + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscoreY, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])
                    agent.save_model('./save_model/'+ agent_name + '_best')
                agent.save_model('./save_model/'+ agent_name)
                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break