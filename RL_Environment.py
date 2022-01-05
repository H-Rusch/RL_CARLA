from ModifiedTensorboard import ModifiedTensorBoard
from Simulator import DEGREE_DIVISOR, WIDTH, HEIGHT

from collections import deque
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Input, Concatenate
from keras.models import Model

import tensorflow as tf
import random
import time
import numpy as np

# ==============================================================================
# -- Defining Constants --------------------------------------------------------
# ==============================================================================

REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "CNN"

DISTANCE_DIVISOR = 500
DISCOUNT = 0.99

START_MODEL = None


# START_MODEL = "models/Xception_____5.40max___-1.75avg___-3.90min__1641311571.model"


# ==============================================================================
# -- DQNAgent ------------------------------------------------------------------
# ==============================================================================

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        """
        Create a neural network which takes an image, the distance and the angle to the next checkpoint and the current
        velocity of the car as an input.
        The output layer has 9 neurons in total. One for each action the vehicle can take.
        """

        if START_MODEL is None:

            # network for image processing
            img_network_in = Input(shape=(HEIGHT, WIDTH, 1), name="img_input")

            img_network = Conv2D(64, (5, 5), padding='same', activation='relu')(img_network_in)
            img_network = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(img_network)

            img_network = Conv2D(128, (5, 5), padding='same', activation='relu')(img_network)
            img_network = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(img_network)

            img_network = Conv2D(256, (3, 3), padding='same', activation='relu')(img_network)
            img_network = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(img_network)

            img_network_out = Flatten()(img_network)

            # network for additional inputs
            add_network_input = Input(shape=(3,), name="add_input")
            add_network_out = Dense(9, activation='relu')(add_network_input)

            # concatenate both networks
            concat_model = Concatenate()([img_network_out, add_network_out])
            concat_model_out = Dense(9, activation='linear')(concat_model)

            model = Model(inputs=[img_network_in, add_network_input], outputs=concat_model_out)
            model.compile(
                loss="categorical_crossentropy",
                optimizer='adam',
                metrics=["accuracy"]
            )

        else:
            model = tf.keras.models.load_model(START_MODEL)

        return model

    def update_replay_memory(self, transition):
        # TODO was ist der Sinn hiervon? (konkret was bedeutet der auskommentierte Code)
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        # TODO das hier aufräumen
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # current
        current_states = [transition[0] for transition in minibatch]
        input_imgs = np.asarray([np.asarray(transition[0]).reshape(480, 640, 1) for transition in current_states])

        input_adds = []
        for i in range(0, len(current_states)):
            input_adds.append([current_states[i][1] / DEGREE_DIVISOR, current_states[i][2] / DISTANCE_DIVISOR,
                               (current_states[i][3] - 50) / 50])

        input_adds = np.asarray(input_adds)

        current_qs_list = self.model.predict_on_batch({"img_input": input_imgs, "add_input": input_adds})

        # new
        new_current_states = [transition[3] for transition in minibatch]
        input_imgs_new = np.asarray(
            [np.asarray(transition[0]).reshape(480, 640, 1) for transition in new_current_states])

        input_adds_new = []
        for i in range(0, len(new_current_states)):
            input_adds_new.append(
                [new_current_states[i][1] / DEGREE_DIVISOR, new_current_states[i][2] / DISTANCE_DIVISOR,
                 (new_current_states[i][3] - 50) / 50])

        input_adds_new = np.asarray(input_adds_new)

        future_qs_list = self.target_model.predict_on_batch({"img_input": input_imgs_new, "add_input": input_adds_new})

        X_img = []
        X_add = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X_img.append((np.array(current_state[0]) / 255).reshape(480, 640, 1))
            X_add.append([(current_state[1] / DEGREE_DIVISOR), (current_state[2] / DISTANCE_DIVISOR),
                          ((current_state[3] - 50) / 50)])
            y.append(current_qs)

        X_img = np.asarray(X_img)
        X_add = np.asarray(X_add)
        y = np.asarray(y)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # with self.graph.as_default():
        self.model.fit({"img_input": X_img, "add_input": X_add}, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        img_in = np.asarray(state[0]) / 255
        add_in = np.asarray([state[1] / DEGREE_DIVISOR, state[2] / DISTANCE_DIVISOR, (state[3] - 50) / 50],
                            dtype=np.float32)
        add_in = add_in.reshape(1, -1)

        qs = self.model.predict(
            x={"img_input": img_in, "add_input": add_in}
        )

        return qs[0]

    def train_in_loop(self):

        # if START_MODEL is None:
        # initialize the neural network with random/ default values
        # img_in = np.random.uniform(size=(1, HEIGHT, WIDTH, 1)).astype(np.float32)
        # add_in = np.asarray([0.0, 0.5, 0.0]).astype(np.float32)
        # add_in = add_in.reshape(1, -1)

        # y = np.random.uniform(size=(1, 9)).astype(np.float32)

        # self.model.fit(x={"img_input": img_in, "add_input": add_in}, y=y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
