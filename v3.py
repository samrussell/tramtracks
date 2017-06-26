# -*- coding: utf-8 -*-

# from https://github.com/keon/deep-q-learning/blob/master/dqn.py
import argparse
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
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
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
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
        act_values = self.model.predict(state)
        action = np.argmax(act_values[0])
        # rewards are 0 or negative so sigmoid gives 0.0 -> 0.5
        # if we double we get 0.0 -> 1.0
        confidence = expit(np.max(act_values[0])) * 2.0
        if np.random.rand() > confidence:
            return random.randrange(self.action_size)
        else:
            return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', help='render display (default false)', default=False, action='store_true')
    parser.add_argument('--load', help='load from file (default false)', default=False, action='store_true')
    parser.add_argument('--train', help='train and save (default false)', default=False, action='store_true')
    commandline_args = parser.parse_args()

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, train=commandline_args.train)
    if commandline_args.load:
        agent.load("./save/cartpole.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            if commandline_args.render:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, time))
                break
        if commandline_args.train:
            if len(agent.memory.memory) > batch_size:
                agent.replay(batch_size)
            if e % 10 == 0:
                agent.save("./save/cartpole.h5")
