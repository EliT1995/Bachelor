import gym
import numpy as np
from skimage import io
import random
from collections import deque
import keras
from keras import backend
from keras.layers import *

env = gym.make("BreakoutDeterministic-v4")
state = env.reset()

ATARI_SHAPE = (84, 84, 4)
ACTION_SIZE = 3

def preprocessing(frame_array):

    #turn to gray
    from skimage.color import rgb2gray

    grayscale_frame = rgb2gray(frame_array)

    # resize
    from skimage.transform import resize
    resized_frame = np.uint8(resize(grayscale_frame, (84, 84), mode='constant') * 255)

    return resized_frame


def epsilon_greedy_policy_action(current_state, episode):
    if np.random.rand() <= epsilon or episode < total_observe_count:
        return random.randrange(ACTION_SIZE)
    else:
        Q_value = model.predict([current_state, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
        return np.argmax(Q_value[0])


def huber_loss(y, q_value):
    error = backend.abs(y - q_value)
    qudratic_part = backend.clip(error, 0.0, 1.0)
    linear_part = error - qudratic_part
    loss = backend.mean(0.5 * backend.square(qudratic_part) + linear_part)
    return loss


def atari_model():
    frames_input = keras.layers.Input(shape=(84, 84, 4), name='frames')
    actions_input = keras.layers.Input((ACTION_SIZE,), name='mask')

    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

    # Architecture of the Model
    first_hidden_layer = keras.layers.convolutional.Conv2D(32, (8, 8), activation='relu', strides=(4, 4))(normalized)
    second_hidden_layer = keras.layers.convolutional.Conv2D(64, (4, 4), activation='relu', strides=(2, 2))(
        first_hidden_layer)
    third_hidden_layer = keras.layers.convolutional.Conv2D(64, (3, 3), activation='relu', strides=(1, 1))(
        second_hidden_layer)
    conv_flattened = keras.layers.core.Flatten()(third_hidden_layer)
    hidden_connected = keras.layers.Dense(512, activation='relu')(conv_flattened)

    output_layer = keras.layers.Dense(ACTION_SIZE)(hidden_connected)

    filtered_output = keras.layers.Multiply(name='QValue')([output_layer, actions_input])

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss=huber_loss)
    return model


replay_memory = deque(maxlen=400000)
model = atari_model()
target_model = atari_model()

nEpisodes = 100000
total_observe_count = 2
epsilon = 1.0
batch_size = 32
gamma = 0.99
final_epsilon = 0.1
epsilon_step_num = 100000
epsilon_decay = (1.0 - final_epsilon) / epsilon_step_num
target_model_change = 100


def get_sample_random_batch_from_replay_memory():
    mini_batch = random.sample(replay_memory, batch_size)

    current_state_batch = np.zeros((batch_size, 84, 84, 4))
    next_state_batch = np.zeros((batch_size, 84, 84, 4))

    actions, rewards, dead = [], [], []

    for idx, val in enumerate(mini_batch):
        current_state_batch[idx] = val[0]
        actions.append(val[1])
        rewards.append(val[2])
        next_state_batch[idx] = val[3]
        dead.append(val[4])

    return current_state_batch, actions, rewards, next_state_batch, dead


def deepQlearn():
    current_state_batch, actions, rewards, next_state_batch, dead = get_sample_random_batch_from_replay_memory()

    action_mask = np.ones((batch_size, ACTION_SIZE))
    #print("action_mask and format si : {}".format(action_mask.shape)) #(32,3)
    #print("next state batch format {}".format(next_state_batch.shape)) #(32,84,84,4)
    next_Q_values = target_model.predict([next_state_batch, action_mask])
    #print("next q values and format si : {}".format(next_Q_values.shape)) #(32,3)
    #print("predict shape is : {}".format([next_state_batch, action_mask]))

    targets = np.zeros((batch_size,)) #(32,)
    print("targets and format si : {}".format(targets.shape))


    for i in range(batch_size):
        if dead[i]:
            targets[i] = -1

        else:
            targets[i] = rewards[i] + gamma * np.amax(next_Q_values[i])

    one_hot_actions = np.eye(ACTION_SIZE)[np.array(actions).reshape(-1)]
    one_hot_targets = one_hot_actions * targets[:, None]

    model.fit([current_state_batch, one_hot_actions], one_hot_targets, epochs=1, batch_size=batch_size, verbose=0)


for episode in range(nEpisodes):

    dead, done, lives_remaining, score = False, False, 5, 0

    current_state = env.reset()
    for _ in range(random.randint(1, 30)):
        current_state, _, _, _ = env.step(1)

    #print(current_state.shape) # (210, 160, 3)
    current_state = preprocessing(current_state)
    #print(current_state.shape) # (84, 84)
    current_state = np.stack((current_state, current_state, current_state, current_state), axis=2)
    #print(current_state.shape) # (84, 84, 4)
    current_state = np.reshape([current_state], (1, 84, 84, 4))

    while not done:
        env.render()
        action = epsilon_greedy_policy_action(current_state, episode)
        real_action = action + 1

        if epsilon > final_epsilon and episode > total_observe_count:
            epsilon -= epsilon_decay

        next_state, reward, done, lives_left = env.step(real_action)

        next_state = preprocessing(next_state)  # 84,84 grayscale frame
        next_state = np.reshape([next_state], (1, 84, 84, 1))
        next_state = np.append(next_state, current_state[:, :, :, :3], axis=3)

        if lives_remaining > lives_left['ale.lives']:
            dead = True
            lives_remaining = lives_left['ale.lives']

        replay_memory.append((current_state, action, reward, next_state, dead))
        if episode > total_observe_count:
            deepQlearn()

            if episode % target_model_change == 0:
                target_model.set_weights(model.get_weights())

        score += reward


        if dead:
            dead = False
        else:
            current_state = next_state

    print("episode: {}, score: {}".format(episode, score))

