# -*- coding: utf-8 -*-

# from https://github.com/keon/deep-q-learning/blob/master/dqn.py
import argparse
import random
import gym
import numpy as np
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam
from scipy.special import expit

EPISODES = 2000
TRIALS_PER_BRAIN = 1
MAX_BRAINS = 40
LEARNING_RATE = 0.1

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.001
        self.brains = []
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(filters=16, kernel_size=(8, 8), strides=(4,4), activation='relu', padding='same', input_shape=self.state_size))
        model.add(Convolution2D(filters=32, kernel_size=(4, 4), strides=(2,2), activation='relu', padding='same'))
        model.add(Convolution2D(filters=32, kernel_size=(4, 4), strides=(2,2), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        brain = model.get_weights()
        self.brains.append(brain)
        return model

    def act(self, state):
        # doesn't always push the fire button
        if np.random.rand() < 0.05:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(self.scale_state(state)))

    def save(self, name):
        # just save the best one
        self.model.set_weights(self.brains[0])
        self.model.save_weights(name)

    def load(self, name):
        # just load the best one
        self.model.load_weights(name)
        self.brains = [self.model.get_weights()]

    def scale_state(self, state):
        return state / 255.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', help='render display (default false)', default=False, action='store_true')
    parser.add_argument('--load', help='load from file (default false)', default=False, action='store_true')
    parser.add_argument('--train', help='train and save (default false)', default=False, action='store_true')
    commandline_args = parser.parse_args()

    env = gym.make('Breakout-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    if commandline_args.load:
        agent.load("./save/breakout.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        # test all the brains
        scored_brains = []
        print("Episode %d" % e)
        for brain in agent.brains:
            score = 0
            agent.model.set_weights(brain)
            for trial in range(TRIALS_PER_BRAIN):
                state = env.reset()
                state = np.reshape(state, (1, ) + state_size)
                while True:
                    if commandline_args.render:
                        env.render()
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    score += reward
                    next_state = np.reshape(next_state, (1, ) + state_size)
                    state = next_state
                    if done:
                        break
            scored_brains.append((brain, score/TRIALS_PER_BRAIN))
            sys.stdout.write('.')
        scores = [x[1] for x in scored_brains]
        average_score = np.array(scores).mean()
        top_5_scores = sorted(scores, reverse=True)[:5]
        print("average score: %d" % average_score)
        print("top 5 scores: %s" % top_5_scores)
        print("mutating")
        if commandline_args.train:
            # sort by top brains
            scored_brains.sort(key=lambda x: x[1], reverse=True)
            # only store first quarter
            new_brains = [x[0] for x in scored_brains[:MAX_BRAINS/4]]
            # make 3 mutations per brain
            mutant_brains = []
            for brain in new_brains:
                for _ in xrange(3):
                    # update every even number layer
                    new_brain = []
                    for index, layer in enumerate(brain):
                        new_layer = layer.copy()
                        if index % 2 == 0:
                            mutation = np.random.random_sample(layer.shape) * LEARNING_RATE - (LEARNING_RATE / 2.0)
                            new_layer += mutation
                        new_brain.append(new_layer)
                    mutant_brains.append(new_brain)
            new_brains += mutant_brains
            agent.brains = new_brains
            agent.save("./save/breakout.h5")
