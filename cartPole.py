import random
import gym
import numpy as np
from collections import deque
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend

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
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        #model = Sequential()

        frames_input = Input(shape=(self.state_size,), name='frames')
        actions_input = Input((self.action_size,), name='mask')

        # Architecture of the Model
        first_hidden_layer = Dense(24, activation='relu')(frames_input)
        second_hidden_layer = Dense(32, activation='relu')(first_hidden_layer)

        output_layer = Dense(self.action_size)(second_hidden_layer)

        filtered_output = Multiply(name='QValue')([output_layer, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def huber_loss(self, y, q_value):
        error = backend.abs(y - q_value)
        qudratic_part = backend.clip(error, 0.0, 1.0)
        linear_part = error - qudratic_part
        loss = backend.mean(0.5 * backend.square(qudratic_part) + linear_part)
        return loss

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        Q_values = self.model.predict([state, np.ones(self.action_size).reshape(1, action_size)])
        return np.argmax(Q_values[0])  # returns action

   # def replay(self, batch_size):
   #     minibatch = random.sample(self.memory, batch_size)

   #     for state, action, reward, next_state, done in minibatch:
   #         target = reward

    #        if not done:
     #           target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
      #      else:
       #         target = -1

        #    target_f = self.model.predict(state)
         #   target_f[0][action] = target
          #  self.model.fit(state, target_f, epochs=1, verbose=0)

     #   if self.epsilon > self.epsilon_min:
      #      self.epsilon *= self.epsilon_decay

    def get_sample_random_batch_from_replay_memory(self):
        mini_batch = random.sample(self.memory, batch_size)

        current_state_batch = np.zeros((batch_size, 4))
        next_state_batch = np.zeros((batch_size, 4))

        actions, rewards, dead = [], [], []

        for idx, val in enumerate(mini_batch):
            current_state_batch[idx] = val[0]
            actions.append(val[1])
            rewards.append(val[2])
            next_state_batch[idx] = val[3]
            dead.append(val[4])

        return current_state_batch, actions, rewards, next_state_batch, dead

    def replay_2(self, batch_size):
        if len(self.memory) < batch_size:
            return

        state, action, reward, next_state, done = self.get_sample_random_batch_from_replay_memory()

        action_mask = np.ones((batch_size, self.action_size))
        next_Q_values = self.target_model.predict([next_state, action_mask])

        targets = np.zeros((batch_size,))

        for i in range(batch_size):
            if done[i]:
                targets[i] = -1

            else:
                targets[i] = reward[i] + self.gamma * np.amax(next_Q_values[i])

        one_hot_actions = np.eye(self.action_size)[np.array(action).reshape(-1)]
        one_hot_targets = one_hot_actions * targets[:, None]

        self.model.fit([state, one_hot_actions], one_hot_targets, epochs=1, batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def set_weights(self):
        self.target_model.set_weights(self.model.get_weights())

if __name__ == "__main__":
    episodes = 1000
    batch_size = 32
    target_model_change = 100

    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Iterate the game
    for episode in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        env.render()
        state = np.reshape(state, [1, state_size])

        for time_t in range(500):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}, score: {}, e: {:.2}"
                      .format(episode, time_t, agent.epsilon))
                break

            # train the agent with the experience of the episode
            agent.replay_2(batch_size)

            if episode % 2 == 0:
                agent.set_weights()
