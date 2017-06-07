#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import random

import gym
import numpy as np
import skimage as skimage
import tensorflow as tf
from keras import backend as K
from skimage import color, transform, exposure

from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer

ACTIONS = 9  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
EXPLORE = 300000.  # frames over which to anneal epsilon
INITIAL_EPSILON = 0.4  # starting value of epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
REPLAY_MEMORY = 100000  # number of previous transitions to remember
BATCH_SIZE = 64  # size of minibatch
LEARNING_RATE = 1e-3
EPISODE_COUNT = 100000
MAX_STEPS = 10000
IMG_ROWS = 64
IMG_COLS = 96
IMG_CHANNELS = 4
INITIALIZE_STDDEV = 0.01

WEIGHT_PATH = '/Developer/Python/AlphaPacman/'

def process_image(image):
    img = skimage.color.rgb2gray(image)
    img = skimage.transform.resize(img, (IMG_ROWS, IMG_COLS), mode='constant')
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
    img = np.array([img])
    img = img.reshape(1, IMG_ROWS, IMG_COLS, 1)
    return img


def train(sess, load_weight):
    env = gym.make('MsPacman-v0')
    buffer = ReplayBuffer(100000)
    agent = DQNAgent(LEARNING_RATE, IMG_ROWS, IMG_COLS, IMG_CHANNELS, INITIALIZE_STDDEV)

    if load_weight:
        print("Now we load weight")
        agent.model.load_weights(WEIGHT_PATH + "model.h5")
        print("Weight load successfully")
    else:
        sess.run(tf.global_variables_initializer())

    epsilon = INITIAL_EPSILON
    # prepare for tensorboard
    r_tfboard = tf.Variable(0.0)
    r_summary = tf.summary.scalar("Reward", r_tfboard)
    summary_writer = tf.summary.FileWriter('/big/MsPacmanLog/reward_log')
    merged_summary_op = tf.summary.merge_all()

    for episode in range(EPISODE_COUNT):
        print("Episode: " + str(episode) + " Replay Buffer " + str(buffer.count()))
        x_t = env.reset()
        x_t = process_image(x_t)
        loss = 0
        total_reward = 0
        step = 0
        life_count = 3
        while step < 80:
            env.render()
            env.step(0)
            step += 1

        # get one channel
        env.render()
        x_t,_,_,_ = env.step(0)
        #print(x_t.shape)
        x_t = skimage.color.rgb2gray(x_t)
        x_t = skimage.transform.resize(x_t, (IMG_ROWS, IMG_COLS), mode='constant')
        x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
        #x_t = np.array([x_t])
        print (x_t.shape)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        s_t = s_t.reshape((1, s_t.shape[0], s_t.shape[1], s_t.shape[2]))

        a_t = np.zeros([ACTIONS])
        if random.random() <= epsilon:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            q = agent.model.predict(s_t)
            action_index = np.argmax(q)
            a_t[action_index] = 1
        
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        while step < MAX_STEPS:
            env.render()
            # take action, observe new state

            x_t1_colored, r_t, terminal, info = env.step(action_index)
            
            # whether be eaten by ghost
            terminal_by_ghost = False
            if life_count > info['ale.lives'] or terminal :
                terminal_by_ghost = True
            life_count = info['ale.lives']

            total_reward += r_t
            x_t1 = process_image(x_t1_colored)
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            # choose new action a_t1 from s_t1 using policy same as Q  
            if_random = False         
            a_t1 = np.zeros([ACTIONS])
            if random.random() <= epsilon:
                if_random = True
                action_index1 = random.randrange(ACTIONS)
                a_t1[action_index1] = 1
            else:
                q = agent.model.predict(s_t1)
                action_index1 = np.argmax(q)
                a_t1[action_index1] = 1

            # reduce the epsilon gradually
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # store the transition in buffer
            buffer.add((s_t, action_index, r_t, s_t1, action_index1, terminal_by_ghost))

            # sample a minibatch to train on
            minibatch = buffer.get_batch(BATCH_SIZE)

            inputs = np.zeros((BATCH_SIZE, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((BATCH_SIZE, ACTIONS))

            # train
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                action_next = minibatch[i][4]
                terminal_t = minibatch[i][5]

                targets[i] = agent.model.predict(state_t)  # Hitting each buttom probability
                q = agent.model.predict(state_t1)
                #print (action_next, q)
                inputs[i] = state_t

                if terminal_t:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * q[0][action_next]

            # targets2 = normalize(targets)
            loss += agent.model.train_on_batch(inputs, targets)

            s_t = s_t1
            step += 1
            # print info
            print("TIMESTEP", step,
                  "/ ACTION", action_index,
                  "/ Next", action_index1,
                  "/ Random", if_random,
                  "/ REWARD", r_t,
                  "/ Loss ", loss,
                  "/ EPSILON", epsilon,
                  "/ eaten", terminal_by_ghost)
            action_index = action_index1
            if terminal or terminal_by_ghost:
                break

        print("************************")
        print("Episode: " + str(episode) + " finished!")
        print("Total reward: ", total_reward)
        print("************************")

        # show reward on tensorboard
        sess.run(tf.assign(r_tfboard, total_reward))
        r_summary = sess.run(merged_summary_op)
        summary_writer.add_summary(r_summary, episode)

        # save progress every 1000 iterations
        if episode % 100 == 0:
            print("Now we save model")
            agent.model.save_weights("model_2.h5", overwrite=True)
            with open("model_2.json", "w") as outfile:
                json.dump(agent.model.to_json(), outfile)


def play():
    env = gym.make('MsPacman-v0')
    agent = DQNAgent(LEARNING_RATE, IMG_ROWS, IMG_COLS, IMG_CHANNELS, INITIALIZE_STDDEV)
    print("Now we load weight")
    agent.model.load_weights(WEIGHT_PATH + "model.h5")
    print("Weight load successfully")
    step = 0
    x_t = env.reset()
    while step < 80:
        env.render()
        env.step(0)
        step += 1
    
    loss = 0
    total_reward = 0
    epsilon = INITIAL_EPSILON
    
    env.reder()
    x_t,_,_,_ = env.step(0)
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (IMG_ROWS, IMG_COLS), mode='constant')
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape((1, s_t.shape[0], s_t.shape[1], s_t.shape[2]))


    for step in range(MAX_STEPS):
        env.render()
        # choose an action epsilon greedy
        a_t = np.zeros([ACTIONS])

        q = agent.model.predict(s_t)
        print("TIMESTEP", step,
              "/ ACTION_PREDICTION", q)
        action_index = np.argmax(q)
        a_t[action_index] = 1

        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal, info = env.step(action_index)
        total_reward += r_t
        x_t1 = process_image(x_t1_colored)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        s_t = s_t1

        # print info
        print("TIMESTEP", step,
              "/ ACTION", action_index,
              "/ REWARD", r_t,
              "/ Loss ", loss,
              "/ EPSILON", epsilon)

        if terminal:
            break
    print("Game ended, Total rewards: " + str(total_reward))


def main(sess):
    parser = argparse.ArgumentParser(description='AlphaPacman')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    parser.add_argument('-l', '--load', action="store_true", help='Whether to load weight or not', required=False)
    args = vars(parser.parse_args())
    if args["mode"] == 'Train':
        train(sess, args["load"])
    else:
        play()


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    main(sess)
