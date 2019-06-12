import random
import gym
import numpy as np
import sys
from collections import deque
from keras.models import *
from keras.layers import Dense
from keras.layers import Multiply
from keras.optimizers import Adam
from keras import initializers
from StatistikLogger import StatistikLogger

multi_step = int(sys.argv[1])

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        frames_input = Input(shape=(self.state_size,), name='frames')
        actions_input = Input((self.action_size,), name='mask')

        initializer = initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None)

        # Architecture of the Model
        first_hidden_layer = Dense(24, kernel_initializer=initializer, bias_initializer='zeros', activation='relu')(frames_input)
        second_hidden_layer = Dense(24, activation='relu')(first_hidden_layer)
        output_layer = Dense(self.action_size)(second_hidden_layer)

        filtered_output = Multiply(name='QValue')([output_layer, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict([state, np.ones(self.action_size).reshape(1, action_size)])
        return np.argmax(act_values[0])  # returns action

    def get_sample_random_batch_from_replay_memory(self):
        mini_batch = random.sample(self.memory, batch_size)

        current_state_batch = np.zeros((batch_size, self.state_size))
        next_state_batch = np.zeros((batch_size, self.state_size))

        actions, rewards, dead = [], [], []

        for idx, val in enumerate(mini_batch):
            current_state_batch[idx] = val[0]
            actions.append(val[1])
            rewards.append(val[2])
            next_state_batch[idx] = val[3]
            dead.append(val[4])

        return current_state_batch, actions, rewards, next_state_batch, dead

    def replay(self, batch_size):
        state, action, reward, next_state, done = self.get_sample_random_batch_from_replay_memory()

        action_mask = np.ones((batch_size, self.action_size))
        next_Q_values = self.target_model.predict([next_state, action_mask])

        targets = np.zeros((batch_size,))

        for i in range(batch_size):
            if done[i]:
                targets[i] = reward[i]

            else:
                targets[i] = reward[i] + self.gamma * np.amax(next_Q_values[i])

        one_hot_actions = np.eye(self.action_size)[np.array(action).reshape(-1)]
        one_hot_targets = one_hot_actions * targets[:, None]

        self.model.fit([state, one_hot_actions], one_hot_targets, epochs=1, batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def set_weights(self):
        self.target_model.set_weights(self.model.get_weights())

if __name__ == "__main__":
    env_name = 'Acrobot-v1'
    env = gym.make(env_name)
    threshold = 195

    score_logger = StatistikLogger('Acrobot-v1_{}multi'.format(multi_step), threshold)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agents = []
    agent = DQNAgent(state_size, action_size)
    agents.append(agent)

    for t in range(1, multi_step):
        agent = DQNAgent(state_size, action_size)
        agent.target_model = agents[t-1].model
        agents.append(agent)

    done = False
    batch_size = 32

    for episode in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        step = 0

        while True:
            #env.render()
            agent = agents[-1]
            action = agents[-1].act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            step += reward

            for t in range(0, multi_step):
                agents[t].remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("Run: {}, exploration: {}, score: {}".format(episode, agents[-1].epsilon, step))
                score_logger.add_score(step, episode)
                break

            if len(agents[-1].memory) > batch_size:
                for t in range(0,multi_step):
                    agents[t].replay(batch_size)

            if step % 50 == 0:
                for t in range(0, multi_step):
                    agents[t].set_weights()

