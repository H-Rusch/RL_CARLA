import random
import time

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Input, Concatenate
from keras.models import Model

import ModifiedTensorboard
from Simulator import DEGREE_DIVISOR, WIDTH, HEIGHT, TARGET_SPEED

# ==============================================================================
# -- Defining Constants --------------------------------------------------------
# ==============================================================================

REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "CNN"

DISTANCE_DIVISOR = 500
DISCOUNT = 0.99


# ==============================================================================
# -- DQNAgent ------------------------------------------------------------------
# ==============================================================================

class DQNAgent:
    def __init__(self, model_name, tensorboard, replay_memory):
        self.model = self.create_model(model_name)
        self.target_model = self.create_model(model_name)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = replay_memory

        self.tensorboard = tensorboard
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self, model_name):
        """
        Create a neural network which takes an image, the distance and the angle to the next checkpoint and the current
        velocity of the car as an input.
        The output layer has 9 neurons in total. One for each action the vehicle can take.
        """

        if model_name is None:
            # network for image processing
            img_network_in = Input(shape=(HEIGHT, WIDTH, 1), name="img_input")

            img_network = Conv2D(32, (5, 5), padding='same', activation="relu")(img_network_in)
            img_network = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(img_network)

            img_network = Conv2D(64, (3, 3), padding='same', activation="relu")(img_network)
            img_network = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(img_network)

            img_network = Conv2D(128, (3, 3), padding='same', activation="relu")(img_network)
            img_network = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(img_network)

            img_network_out = Flatten()(img_network)
            img_network_out = Dense(256, activation='relu')(img_network_out)
            img_network_out = Dense(9)(img_network_out)
            # network for additional inputs
            add_network_input = Input(shape=(3,), name="add_input")
            add_network_out = Dense(9, activation='relu')(add_network_input)

            # concatenate both networks
            concat_model = Concatenate()([img_network_out, add_network_out])
            concat_model_out = Dense(9, activation='linear')(concat_model)

            model = Model(inputs=[img_network_in, add_network_input], outputs=concat_model_out)
            model.compile(
                loss="mse",
                optimizer='adam',
                metrics=["accuracy"]
            )

        else:
            model = tf.keras.models.load_model(model_name)

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        """
        The images and additional information are extracted from a list of states randomly selected from the replay
        memory. With these inputs a prediction is made. Afterwards the same thing is done for the new states which
        follow the original states after the selected action was taken.
        The maximum future q values are calculated for each state and based on that the Neural Networks are updated.
        """
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # a list of states randomly selected from the replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        input_images, input_adds = [], []
        input_images_new, input_adds_new = [], []
        for (current_state, _, _, new_state, _) in minibatch:
            # get image and additional information from current states
            input_images.append(np.asarray(np.asarray(current_state[0]) / 255).reshape(480, 640))
            input_adds.append([current_state[1] / DEGREE_DIVISOR, current_state[2] / DISTANCE_DIVISOR,
                               (current_state[3] - TARGET_SPEED) / TARGET_SPEED])

            # get image and additional information from states following the current states
            input_images_new.append(np.asarray(np.asarray(new_state[0]) / 255).reshape(480, 640))
            input_adds_new.append([new_state[1] / DEGREE_DIVISOR, new_state[2] / DISTANCE_DIVISOR,
                                   (new_state[3] - TARGET_SPEED) / TARGET_SPEED])

        input_images = np.asarray(input_images)
        input_adds = np.asarray(input_adds)

        input_images_new = np.asarray(input_images_new)
        input_adds_new = np.asarray(input_adds_new)

        current_qs_list = self.model.predict({"img_input": input_images, "add_input": input_adds},
                                             PREDICTION_BATCH_SIZE)
        future_qs_list = self.target_model.predict({"img_input": input_images_new, "add_input": input_adds_new},
                                                   PREDICTION_BATCH_SIZE)

        # calculate maximum future q values for states
        y = []
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            y.append(current_qs)
        y = np.asarray(y)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        # update the weights in the Neural Network
        self.model.fit({"img_input": input_images, "add_input": input_adds}, np.array(y),
                       batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        # update target model
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        """
        Predict the Q-values for the current state given as an input. The state will be normalized in oder to get values
        between 0 and 1 as inputs for the Neural Network.
        """
        img_in = np.asarray(state[0]) / 255
        add_in = np.asarray(
            [state[1] / DEGREE_DIVISOR, state[2] / DISTANCE_DIVISOR, (state[3] - TARGET_SPEED) / TARGET_SPEED],
            dtype=np.float32)
        add_in = add_in.reshape(1, -1)

        qs = self.model.predict(
            x={"img_input": img_in, "add_input": add_in}
        )

        return qs[0]

    def train_in_loop(self):
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
