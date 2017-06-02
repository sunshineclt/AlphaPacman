#!/usr/bin/env python
from __future__ import print_function

import argparse

import gym
import json
import random
import tensorflow as tf
from keras import backend as K
import numpy as np
from skimage import transform, color, exposure
import skimage as skimage

from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer

ACTIONS = 9  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
EXPLORE = 3000000.  # frames over which to anneal epsilon
INITIAL_EPSILON = 0.3  # starting value of epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
REPLAY_MEMORY = 10000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
LEARNING_RATE = 1e-3
EPISODE_COUNT = 100000
MAX_STEPS = 10000
IMG_ROWS = 80
IMG_COLS = 80
IMG_CHANNELS = 1
INITIALIZE_STDDEV = 0.01

WEIGHT_PATH = '/Developer/Python/AlphaPacman/'


def process_image(img):
    img = skimage.color.rgb2gray(img)
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

    for episode in range(EPISODE_COUNT):
        print("Episode: " + str(episode) + " Replay Buffer " + str(buffer.count()))
        s_t = env.reset()
        s_t = process_image(s_t)
        loss = 0
        total_reward = 0

        for step in range(MAX_STEPS):
            env.render()
            # choose an action epsilon greedy
            a_t = np.zeros([ACTIONS])
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = agent.model.predict(s_t)
                action_index = np.argmax(q)
                a_t[action_index] = 1

            # reduce the epsilon gradually
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observed next state and reward
            s_t1_colored, r_t, terminal, info = env.step(action_index)
            total_reward += r_t
            s_t1 = process_image(s_t1_colored)

            # store the transition in buffer
            buffer.add((s_t, action_index, r_t, s_t1, terminal))

            # sample a minibatch to train on
            minibatch = buffer.get_batch(BATCH_SIZE)

            inputs = np.zeros((BATCH_SIZE, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 80, 80, 4
            targets = np.zeros((BATCH_SIZE, ACTIONS))

            # train
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal_t = minibatch[i][4]

                targets[i] = agent.model.predict(state_t)  # Hitting each buttom probability
                q = agent.model.predict(state_t1)

                if terminal_t:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(q)

            # targets2 = normalize(targets)
            loss += agent.model.train_on_batch(inputs, targets)

            s_t = s_t1

            # print info
            print("TIMESTEP", step,
                  "/ ACTION", action_index,
                  "/ REWARD", r_t,
                  "/ Loss ", loss,
                  "/ EPSILON", epsilon)

            if terminal:
                print("************************")
                print("Terminal is true!")
                print("************************")
                break

        print("************************")
        print("Episode: " + str(episode) + " finished!")
        print("Total reward: ", total_reward)
        print("************************")

        # save progress every 1000 iterations
        if episode % 100 == 0:
            print("Now we save model")
            agent.model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(agent.model.to_json(), outfile)


def play():
    env = gym.make('MsPacman-v0')
    agent = DQNAgent(LEARNING_RATE, IMG_ROWS, IMG_COLS, IMG_CHANNELS, INITIALIZE_STDDEV)
    print("Now we load weight")
    agent.model.load_weights(WEIGHT_PATH + "model.h5")
    print("Weight load successfully")

    s_t = env.reset()
    s_t = process_image(s_t)
    loss = 0
    total_reward = 0
    epsilon = INITIAL_EPSILON
    for step in range(MAX_STEPS):
        env.render()
        # choose an action epsilon greedy
        a_t = np.zeros([ACTIONS])
        q = agent.model.predict(s_t)
        action_index = np.argmax(q)
        a_t[action_index] = 1

        # run the selected action and observed next state and reward
        s_t1_colored, r_t, terminal, info = env.step(action_index)
        total_reward += r_t
        s_t1 = process_image(s_t1_colored)
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
