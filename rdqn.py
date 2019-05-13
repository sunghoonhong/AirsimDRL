'''
Author: Sunghoon Hong
Title: RDQNAgent.py
Descr
    Recurrent Deep Q-Network Agent for Airsim,
Detail:
    - not use join()
    - reset for zero-image error
    - tensorflow v1 + keras
    - hard update for target model

'''


import os
import csv
import time
import random
import argparse
from copy import deepcopy
from collections import deque
from datetime import datetime as dt
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import TimeDistributed, BatchNormalization, Flatten, Lambda, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU, Activation
from keras.optimizers import Adam
from keras.models import Model
from PIL import Image
import cv2
from airsim_env import Env, ACTION

np.set_printoptions(suppress=True, precision=4)
agent_name = 'rdqn'


class RDQNAgent(object):
    
    def __init__(self, state_size, action_size, lr,
                gamma, batch_size, memory_size, 
                epsilon, epsilon_end, decay_step, load_model):
        self.state_size = state_size
        self.vel_size = 3
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_step = decay_step
        self.epsilon_decay = (epsilon - epsilon_end) / decay_step

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.critic = self.build_model()
        self.target_critic = self.build_model()
        self.critic_update = self.build_critic_optimizer()
        self.sess.run(tf.global_variables_initializer())
        if load_model:
            self.load_model('./save_model/'+ agent_name)
        
        self.target_critic.set_weights(self.critic.get_weights())

        self.memory = deque(maxlen=self.memory_size)

    def build_model(self):
        # image process
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

        # state process
        # state_process = Concatenate()([image_process, vel_process])
        state_process = image_process

        # Critic
        Qvalue = Dense(128, kernel_initializer='he_normal', use_bias=False)(state_process)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(128, kernel_initializer='he_normal', use_bias=False)(Qvalue)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue)
        critic = Model(inputs=[image, vel], outputs=Qvalue)

        critic._make_predict_function()
        
        return critic

    def build_critic_optimizer(self):
        action = K.placeholder(shape=(None, ), dtype='int32')
        y = K.placeholder(shape=(None, ), dtype='float32')
        pred = self.critic.output
        
        # loss = K.mean(K.square(pred - y))
        # Huber Loss
        action_vec = K.one_hot(action, self.action_size)
        Q = K.sum(pred * action_vec, axis=1)
        error = K.abs(y - Q)
        quadratic = K.clip(error, 0.0, 1.0)
        linear = error - quadratic
        loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function(
            [self.critic.input[0], self.critic.input[1], action, y],
            [loss],
            updates=updates
        )
        return train

    def get_action(self, state):
        Qs = self.critic.predict(state)[0]
        Qmax = np.amax(Qs)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size), np.argmax(Qs), Qmax
        return np.argmax(Qs), np.argmax(Qs), Qmax

    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)

        images = np.zeros([self.batch_size] + self.state_size)
        vels = np.zeros([self.batch_size, self.vel_size])
        actions = np.zeros((self.batch_size))
        rewards = np.zeros((self.batch_size))
        next_images = np.zeros([self.batch_size] + self.state_size)
        next_vels = np.zeros([self.batch_size, self.vel_size])
        dones = np.zeros((self.batch_size))

        targets = np.zeros((self.batch_size, 1))
        
        for i, sample in enumerate(batch):
            images[i], vels[i] = sample[0]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_images[i], next_vels[i] = sample[3]
            dones[i] = sample[4]
        states = [images, vels]
        next_states = [next_images, next_vels]
        target_next_Qs = self.target_critic.predict(next_states)
        targets = rewards + self.gamma * (1 - dones) * np.amax(target_next_Qs, axis=1)
        critic_loss = self.critic_update(states + [actions, targets])
        return critic_loss[0]

    def append_memory(self, state, action, reward, next_state, done):        
        self.memory.append((state, action, reward, next_state, done))
        
    def load_model(self, name):
        if os.path.exists(name + '.h5'):
            self.critic.load_weights(name + '.h5')
            print('Model loaded')

    def save_model(self, name):
        self.critic.save_weights(name + '.h5')

    def update_target_model(self):
        self.target_critic.set_weights(self.critic.get_weights())


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
    scaling_factor = 1.0
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

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--play',       action='store_true')
    parser.add_argument('--img_height', type=int,   default=72)
    parser.add_argument('--img_width',  type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--seqsize',    type=int,   default=5)
    parser.add_argument('--epoch',      type=int,   default=1)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--memory_size',type=int,   default=50000)
    parser.add_argument('--train_start',type=int,   default=3000)
    parser.add_argument('--train_rate', type=int,   default=5)
    parser.add_argument('--target_rate',type=int,   default=1000)
    parser.add_argument('--epsilon',    type=float, default=1)
    parser.add_argument('--epsilon_end',type=float, default=0.05)
    parser.add_argument('--decay_step', type=int,   default=20000)

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
    agent = RDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model
    )

    episode = 0
    env = Env()

    if args.play:
        while True:
            try:
                done = False
                bug = False

                # stats
                bestY, timestep, score, avgQ = 0., 0, 0., 0.

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
                    Qs = agent.critic.predict(state)[0]
                    action = np.argmax(Qs)
                    Qmax = np.amax(Qs)
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
                    avgQ += float(Qmax)
                    score += reward
                    if info['Y'] > bestY:
                        bestY = info['Y']
                    print('%s' % (ACTION[action]), end='\r', flush=True)

                    if args.verbose:
                        print('Step %d Action %s Reward %.2f Info %s:' % (timestep, real_action, reward, info['status']))

                    state = next_state

                if bug:
                    continue
                
                avgQ /= timestep

                # done
                print('Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f'
                        % (episode, bestY, timestep, score, avgQ))

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
        global_train_num = 0
        while True:
            try:
                done = False
                bug = False

                # stats
                bestY, timestep, score, avgQ = 0., 0, 0., 0.
                train_num, loss = 0, 0.

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
                    timestep += 1
                    global_step += 1
                    if len(agent.memory) >= args.train_start and global_step >= args.train_rate:
                        for _ in range(args.epoch):
                            c_loss = agent.train_model()
                            loss += float(c_loss)
                            train_num += 1
                            global_train_num += 1
                        global_step = 0
                    if global_train_num >= args.target_rate:
                        agent.update_target_model()
                        global_train_num = 0

                    action, policy, Qmax = agent.get_action(state)
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
                    agent.append_memory(state, action, reward, next_state, done)

                    # stats
                    avgQ += float(Qmax)
                    score += reward
                    if info['Y'] > bestY:
                        bestY = info['Y']

                    print('%s | %s' % (ACTION[action], ACTION[policy]), end='\r', flush=True)


                    if args.verbose:
                        print('Step %d Action %s Reward %.2f Info %s:' % (timestep, real_action, reward, info['status']))

                    state = next_state

                    if agent.epsilon > agent.epsilon_end:
                        agent.epsilon -= agent.epsilon_decay

                if bug:
                    continue
                if train_num:
                    loss /= train_num
                avgQ /= timestep

                # done
                if args.verbose or episode % 10 == 0:
                    print('Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f'
                            % (episode, bestY, timestep, score, avgQ))
                stats = [
                    episode, timestep, score, bestY, \
                    loss, info['level'], avgQ, info['status']
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