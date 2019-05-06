import random
import gym
import numpy as np
from collections import deque
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from StatistikLogger import StatistikLogger

multiStep = 5

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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

    def remember(self, state, action, reward, timeStep, next_state, done):
        self.memory.append((state, action, reward, timeStep, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        Q_values = self.model.predict(state)
        return np.argmax(Q_values[0])  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, timeStep, next_state, done in minibatch:
            target = -1

            done1 = self.get_next_state_done(timeStep)
            next_state = self.get_next_state(timeStep)

            if done1:
                target = self.discount(timeStep)

            if not done:
                target = reward + (self.gamma**multiStep) * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def discount(self, timeStep):
        #Compute the gamma-discounted rewards over an episode
        rewards = []
        discounted_rewards = 0

        for elem in self.memory:
            if timeStep <= elem[3] <= timeStep + (multiStep - 1):
                rewards.append(elem[2])

        for t in range(0, len(rewards)):
            discounted_rewards = rewards[t] + discounted_rewards * self.gamma

        return discounted_rewards

    def get_next_state(self, timeStep):
        elements = []

        for elem in self.memory:
            if timeStep <= elem[3] <= timeStep + (multiStep - 1):
                elements.append(elem)

        for t in range(0, len(elements)):
            element = elements[t]
            if element[5] is True:
                return element[4]

        element = elements[len(elements) - 1]
        return element[4]

    def get_next_state_done(self, timeStep):
        elements = []

        for elem in self.memory:
            if timeStep <= elem[3] <= timeStep + (multiStep - 1):
                elements.append(elem)

        for t in range(0, len(elements)):
            element = elements[t]
            if element[5] is True:
                return element[5]

        element = elements[len(elements) - 1]
        return element[5]


if __name__ == "__main__":
    episodes = 1000
    batch_size = 32
    target_model_change = 100

    env = gym.make('CartPole-v0')
    score_logger = StatistikLogger('CartPole-v0_new', 195)


    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Iterate the game
    timeStep = -1

    for episode in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        #env.render()
        state = np.reshape(state, [1, state_size])

        for time_t in range(500):
            timeStep += 1
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, timeStep, next_state, done)

            state = next_state

            if done:
                print("Run: {}, exploration: {}, score: {}".format(episode, agent.epsilon, time_t))
                score_logger.add_score(time_t, episode)
                break

            # train the agent with the experience of the episode
            agent.replay(batch_size)
