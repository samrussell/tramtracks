# -*- coding: utf-8 -*-

# from https://github.com/keon/deep-q-learning/blob/master/dqn.py
import argparse
import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Dropout
from keras.optimizers import Adam
from scipy.special import expit

EPISODES = 2000

class Memory:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = []

    def is_full(self):
        return len(self.memory) == self.mem_size

    def remember(self, value, remember_chance):
        if np.random.rand() > remember_chance:
            return

        if self.is_full():
            self.memory[np.random.choice(self.mem_size)] = value
        else:
            self.memory.append(value)


class DQNAgent:
    def __init__(self, state_size, action_size, train=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(2000)
        self.gamma = 0.99    # discount rate
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(filters=16, kernel_size=(8, 8), strides=(4,4), activation='relu', padding='same', input_shape=self.state_size))
        model.add(Convolution2D(filters=32, kernel_size=(4, 4), strides=(2,2), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        # this gives us a number,
        # 0 in (reward) gives 0 out
        # +/- 1 in (reward) gives 0.25 out
        # +/- 10 in (reward) gives 0.95 out
        # this will help us remember big rewards and not worry too much about smaller ones
        remember_chance = np.log(np.abs(reward * 0.8) + 1) / np.log(10)
        self.memory.remember((state, action, reward, next_state, done), remember_chance)

    def act(self, state):
        act_values = self.model.predict(self.scale_state(state))
        action = np.argmax(act_values[0])
        # take difference between max confidence and mean confidence
        max_confidence = np.max(act_values[0])
        mean_confidence = np.mean(act_values[0])
        vote_confidence = expit(max_confidence - mean_confidence)
        #print act_values[0]
        #print vote_confidence
        if np.random.rand() > vote_confidence:
            return random.randrange(self.action_size)
        else:
            return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(self.scale_state(next_state))[0])
            target_f = self.model.predict(self.scale_state(state))
            target_f[0][action] = target
            self.model.fit(self.scale_state(state), target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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
    agent = DQNAgent(state_size, action_size, train=commandline_args.train)
    if commandline_args.load:
        agent.load("./save/breakout.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, ) + state_size)
        points = 0
        for time in range(500):
            if commandline_args.render:
                env.render()
            action = agent.act(state)
            #print("action %d" % action)
            next_state, reward, done, _ = env.step(action)
            points += reward
            reward = reward if not done else -10
            next_state = np.reshape(next_state, (1, ) + state_size)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, points))
                break
        if commandline_args.train:
            if len(agent.memory.memory) > batch_size:
                agent.replay(batch_size)
            if e % 10 == 0:
                agent.save("./save/breakout.h5")
