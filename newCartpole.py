import random
import gym
import numpy as np
from collections import deque
from keras.models import *
from keras.layers import Dense
from keras.layers import Multiply
from keras import initializers
from keras.optimizers import Adam
from StatistikLogger import StatistikLogger

multi_step = 3

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
        self.eli = []
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        frames_input = Input(shape=(self.state_size,), name='frames')
        actions_input = Input((self.action_size,), name='mask')

        initializer = initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None)

        # Architecture of the Model
        first_hidden_layer = Dense(24, kernel_initializer=initializer, bias_initializer='zeros', activation='relu')(
            frames_input)
        second_hidden_layer = Dense(24, activation='relu')(first_hidden_layer)
        output_layer = Dense(self.action_size)(second_hidden_layer)

        filtered_output = Multiply(name='QValue')([output_layer, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

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

        act_values = self.model.predict([state, np.ones(self.action_size).reshape(1, action_size)])
        return np.argmax(act_values[0])  # returns action

    def get_sample_random_batch_from_replay_memory(self):
        mini_batch = random.sample(self.memory, batch_size)

        current_state_batch = np.zeros((batch_size, 4))
        next_state_batch = np.zeros((batch_size, 4))

        actions, rewards, done = [], [], []

        for idx, val in enumerate(mini_batch):
            current_state_batch[idx] = val[0]
            actions.append(val[1])
            rewards.append(val[2])
            next_state_batch[idx] = val[3]
            done.append(val[4])

        return current_state_batch, actions, rewards, next_state_batch, done

    def replay(self, batch_size):
        state, action, reward, next_state, done = self.get_sample_random_batch_from_replay_memory()

        action_mask = np.ones((batch_size, self.action_size))
        targets = np.zeros((batch_size,))

        next_Q_values = self.target_model.predict([next_state, action_mask])

        for i in range(batch_size):
            if done[i]:
                targets[i] = reward[i]

            else:
                targets[i] = reward[i] + self.gamma**multi_step * np.amax(next_Q_values[i])

        one_hot_actions = np.eye(self.action_size)[np.array(action).reshape(-1)]
        one_hot_targets = one_hot_actions * targets[:, None]

        self.model.fit([state, one_hot_actions], one_hot_targets, epochs=1, batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def set_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def discount(self, experiences):
        # Compute the gamma-discounted rewards over an episode
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
    threshold = 195

    score_logger = StatistikLogger('CartPole-v0_new', threshold)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32

    previous_experiences= deque(maxlen=multi_step)

    for episode in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        step = 0

        while True:
            step += 1
            # env.render()
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            previous_experiences.append((state, action, reward, next_state, done))

            if len(previous_experiences) >= multi_step:
                agent.remember(previous_experiences)

            state = next_state

            if done:
                #print("Run: {}, exploration: {}, score: {}".format(episode, agent.epsilon, step))
                score_logger.add_score(step, episode)
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if step % 8 == 0:
                agent.set_weights()

