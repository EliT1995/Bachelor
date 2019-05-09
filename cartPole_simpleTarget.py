import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from StatistikLogger import StatistikLogger

EPISODES = 1000

multiStep = 200

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
        self.tau = .125
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, experiences):
        state = experiences[0][0]
        action = experiences[0][1]
        reward = self.discount(experiences)
        next_state, done = self.get_next_state(experiences)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward

            else:
                target = (reward + self.gamma**multiStep *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def set_weights(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
            self.target_model.set_weights(target_weights)


    def discount(self, experiences):
        #Compute the gamma-discounted rewards over an episode
        discounted_rewards = 0
        t = 0

        for state, action, reward, next_state, done in experiences:
            discounted_rewards += reward * self.gamma ** t
            t += 1
            if done:
                break

        return discounted_rewards

    def get_next_state(self, experiences):

        n_step_next_state = []
        n_step_done = False
        for state, action, reward, next_state, done in experiences:
            n_step_next_state = next_state
            n_step_done = done

            if done:
                break

        return n_step_next_state, n_step_done


if __name__ == "__main__":
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    score_logger = StatistikLogger('CartPole-v0_simple', 195)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32

    run = 0

    for run in range(500):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        step = 0
        previous_experiences = deque(maxlen=multiStep)

        while True:
            step += 1
            # env.render()
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            previous_experiences.append((state, action, reward, next_state, done))

            if len(previous_experiences) >= multiStep or done:
                agent.remember(previous_experiences)

            state = next_state

            if done:
                # print("Run: {}, exploration: {}, score: {}".format(run, agent.epsilon, step))
                score_logger.add_score(step, run)
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            agent.set_weights()

