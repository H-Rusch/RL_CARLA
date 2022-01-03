from __future__ import print_function
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# TODO testen, ob bei den Schwarz/ weiß bildern die Werte zwischen 0 und 1 sind. Wenn ja müssen wir die Bilder nicht mehr durch 255 teilen

import glob
import logging
import math
import sys

import random
import time
import numpy.random as numpy_random
import numpy as np
import cv2
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, AveragePooling2D, Flatten, Input, Concatenate
from keras.optimizer_v2.adam import Adam
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard

import tensorflow as tf
from threading import Thread

from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import cv2
except ImportError:
    raise RuntimeError('cannot import opencv, make sure opencv-python package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc, Transform, Location, Rotation

# ==============================================================================
# -- Defining Constants --------------------------------------------------------
# ==============================================================================

PORT = "127.0.0.1"
HOST = 2000

WIDTH = 640
HEIGHT = 480
FOV = 110

SHOW_IMAGE = False
SPAWN_LOCATION = (79.19, 302.39, 2.0)

DEGREE_DIVISOR = 360
DISTANCE_DIVISOR = 500

SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -1

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95  ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


# ==============================================================================
# -- ModifiedTensorBoard ---------------------------------------------------------------
# ==============================================================================

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    # Overwritten, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overwritten
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overwritten, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


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

        current_states = [transition[0] for transition in minibatch]
        np.array(current_states[0]) / 255
        current_states[1] /= DEGREE_DIVISOR
        current_states[2] /= DISTANCE_DIVISOR
        current_states[3] = (current_states[3] - 50) / 50

        # with self.graph.as_default():
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = [transition[3] for transition in minibatch]
        np.array(new_current_states[0]) / 255
        new_current_states[1] /= DEGREE_DIVISOR
        new_current_states[2] /= DISTANCE_DIVISOR
        new_current_states[3] = (new_current_states[3] - 50) / 50

        # with self.graph.as_default():
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append((np.array(current_state[0]) / 255))
            X.append([(current_state[1] / DEGREE_DIVISOR), (current_state[2] / DISTANCE_DIVISOR),
                      ((current_state[3] - 50) / 50)])
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # with self.graph.as_default():
        self.model.fit(X, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
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
        # initialize the neural network with random/ default values
        img_in = np.random.uniform(size=(1, HEIGHT, WIDTH, 1)).astype(np.float32)
        add_in = np.asarray([0.0, 0.5, 0.0]).astype(np.float32)
        add_in = add_in.reshape(1, -1)

        y = np.random.uniform(size=(1, 9)).astype(np.float32)

        self.model.fit(x={"img_input": img_in, "add_input": add_in}, y=y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


# TODO car environment und das was damit zusammenhängt in eigene Datei auslagern (?)
# ==============================================================================
# -- CarEnvironment ---------------------------------------------------------------
# ==============================================================================

class CarEnvironment(object):
    """
    Class representing the surrounding environment in the CARLA simulator.
    Contains the needed sensors and the vehicle.
    """

    def __init__(self, carla_world, checkpoint_manager: "CheckpointManager"):
        """Constructor method"""
        self.world = carla_world
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle = Vehicle(self)

        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None

        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_manager.init_checkpoints()

        self.episode_start = None
        self.extra_time = None

        self.checkpoint_distance = 10_000

    def restart(self):
        """
        Restart the world by cleaning up old sensors and spawning them again.
        :returns: The current state the environment
        """

        # clean up old objects
        self.destroy()

        self.checkpoint_manager.reset()

        # spawn the actor
        self.vehicle.spawn_actor()

        # Set up and spawn the sensors.
        self.camera_manager = CameraManager(self.vehicle.actor, show_image=SHOW_IMAGE)
        self.collision_sensor = CollisionSensor(self.vehicle.actor)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle.actor)

        self.camera_manager.spawn_cameras()

        while self.camera_manager.lane_detection_img is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        return self.get_state()

    def step(self, action: int) -> tuple:
        """
        Execute a step in the car environment. The actor will be rewarded based on the action he took.
        If the actor collides into something, the episode should be ended and he is severely punished.
        If the actor is going under 50 kmh the actor is mildly punished, but if the actor is going over 50 kmh he is
        mildly rewarded instead.

        :param action: the action the car should perform, selected by the neural network
        :return: a tuple of (current state the actor is in; the reward earned in this step; an information whether the
        episode is over; No additional information)
        """
        reward = 0
        done = False

        self.vehicle.execute_action(action)

        car_location = (self.vehicle.actor.get_location().x, self.vehicle.actor.get_location().y)
        if self.checkpoint_manager.check_in_current(car_location):
            reward += 1

            # give extra reward based on how much time was left when reaching the checkpoint
            time_left = (self.episode_start + SECONDS_PER_EPISODE + self.extra_time) - time.time()
            if time_left > 0:
                reward += time_left / 10

            self.checkpoint_manager.toggle_next()

            self.extra_time += 10
            self.checkpoint_distance = 10_000

            # self.world.debug.draw_point(
            #    self.checkpoint_manager.checkpoints[self.checkpoint_manager.current].get_location(),
            #    size=1.0,
            #    color=carla.Color(r=255, g=0, b=0),
            #    life_time=5.0)

            if self.checkpoint_manager.check_finished():
                reward += 100
                done = True

        if not done:
            kmh = self.vehicle.get_kmh()

            if len(self.collision_sensor.history) != 0:
                done = True
                reward -= 3
            elif kmh < 35:
                done = False
                reward -= 0.1
            else:
                done = False
                reward += 0.1

            # stop episode if it takes too long
            if time.time() > self.episode_start + SECONDS_PER_EPISODE + self.extra_time:
                done = True

        current_state = self.get_state()

        # give extra reward if distance to checkpoint is lowered
        distance = current_state[2]
        if distance < self.checkpoint_distance:
            self.checkpoint_distance = distance
            reward += 0.1

        return current_state, reward, done, None

    def get_state(self):
        # current camera image
        img = self.camera_manager.lane_detection_img

        # angle and distance to the next checkpoint
        distance, angle = self.get_next_checkpoint_state()

        kmh = self.vehicle.get_kmh()

        return [img], angle, distance, kmh

    def get_next_checkpoint_state(self) -> tuple:
        vehicle_transform = self.vehicle.actor.get_transform()

        checkpoint = self.checkpoint_manager.checkpoints[self.checkpoint_manager.current]
        checkpoint_location = checkpoint.get_location()

        distance = int(vehicle_transform.location.distance(checkpoint_location))

        # calculate the angle between the car and the checkpoint by computing the atan2 and
        # normalizing the angle to a value in [0, 360)
        # 1 is one ° to the left, 359 is one ° to the right
        c_x, c_y = checkpoint_location.x, checkpoint_location.y
        v_x, v_y = vehicle_transform.location.x, vehicle_transform.location.y

        raw_angle = math.atan2(c_y - v_y, c_x - v_x)
        raw_angle = math.degrees(raw_angle)

        angle = int((raw_angle - vehicle_transform.rotation.yaw) % DEGREE_DIVISOR)

        return distance, angle

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager,
            self.collision_sensor,
            self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()
                time.sleep(0.01)


# ==============================================================================
# -- Vehicle ------------------------------------------------------------------
# ==============================================================================

class Vehicle:
    """Class representing the vehicle in the simulation. """

    def __init__(self, world: CarEnvironment):
        self.actor = None
        self.world = world

        # the cars model
        self.blueprint = numpy_random.choice(self.world.world.get_blueprint_library().filter("model3"))
        self.blueprint.set_attribute('role_name', 'hero')

        # spawn point of the car
        x, y, z = SPAWN_LOCATION
        self.spawn_point = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=180))

    def spawn_actor(self):
        try:
            self.actor = self.world.world.spawn_actor(self.blueprint, self.spawn_point)
        except RuntimeError:
            print("Runtime Error")

    def execute_action(self, action):
        """
        Execute the selected action by addressing the actuators.
        Supports going left/ straight/ right while accelerating/ braking/ keeping a steady speed.
        """
        throttle_value = None
        steer_value = None
        brake_value = None

        # accelerate
        if action in [0, 1, 2]:
            throttle_value = 1.0
            brake_value = 0.0

            # go left
            if action == 0:
                steer_value = -1.0
            # go straight
            elif action == 1:
                steer_value = 0
            # go right
            else:
                steer_value = 1.0

        # steady speed
        elif action in [3, 4, 5]:
            throttle_value = 0.0
            brake_value = 0.0

            # go left
            if action == 3:
                steer_value = -1.0
            # go straight
            elif action == 4:
                steer_value = 0
            # go right
            else:
                steer_value = 1.0

        elif action in [6, 7, 8]:
            throttle_value = 0.0
            brake_value = 1.0

            # go left
            if action == 6:
                steer_value = -1.0
            # go straight
            elif action == 7:
                steer_value = 0
            # go right
            else:
                steer_value = 1.0

        self.actor.apply_control(carla.VehicleControl(throttle=throttle_value, steer=steer_value, brake=brake_value))
        time.sleep(0.05)  # maybe?

    def destroy(self):
        if self.actor is not None:
            self.actor.destroy()
        self.actor = None

    def get_kmh(self) -> int:
        v = self.actor.get_velocity()
        return int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, show_image: bool = False):
        """
        Constructor Method
        :param parent_actor: the actor the cameras are attached to
        :param show_image: show the image the camera records
        """
        self.rgb_camera = None
        self.sem_seg_camera = None

        self.lane_detection_img = None

        self.show = show_image
        self._parent = parent_actor

        # transforms for camera
        # first person
        self._camera_transform_fp = (carla.Transform(
            carla.Location(x=2, z=1)), carla.AttachmentType.Rigid)
        # third person
        self._camera_transform_tp = (carla.Transform(
            carla.Location(x=-5.5, z=2.5)), carla.AttachmentType.Rigid)

        # sensors for this camera manager
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)']]

        # get the blueprints for the sensors
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(WIDTH))
            bp.set_attribute('image_size_y', str(HEIGHT))
            bp.set_attribute('fov', str(FOV))

            item.append(bp)

    def destroy(self):
        """Destroy the sensors in the simulation."""
        if self.sem_seg_camera is not None:
            self.sem_seg_camera.destroy()
        self.sem_seg_camera = None

        if self.rgb_camera is not None:
            self.rgb_camera.destroy()
        self.rgb_camera = None

    def spawn_cameras(self):
        """
        Spawn the cameras in the simulation.
        Spawns only the semantic segmentation camera in normal mode and an additional rgb camera in debug mode
        """
        if self.show:
            self.rgb_camera = self._parent.get_world() \
                .spawn_actor(self.sensors[0][-1],
                             self._camera_transform_tp[0],
                             attach_to=self._parent,
                             attachment_type=self._camera_transform_tp[-1])
            self.rgb_camera.listen(lambda image: self.show_rgb_image(image))

        self.sem_seg_camera = self._parent.get_world() \
            .spawn_actor(self.sensors[1][-1],
                         self._camera_transform_fp[0],
                         attach_to=self._parent,
                         attachment_type=self._camera_transform_fp[-1])
        self.sem_seg_camera.listen(lambda image: self.parse_image(image))

    def show_rgb_image(self, image):
        """Show the RGB image of the actor in its environment in a separate window. """
        image.convert(self.sensors[0][1])
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]

        cv2.imshow("", img)
        cv2.waitKey(0)

    def parse_image(self, image):
        """Parse the semantic segmentation image into the lane detection image."""
        image.convert(self.sensors[1][1])
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]

        img = lane_detection_from_sem_seg(img)
        self.lane_detection_img = img


def lane_detection_from_sem_seg(img):
    """
    Convert a semantic segmentation image into a black and white image where only the contours of the road are highlighted.
    :param img: the image returned form the semantic segmentation camera
    :return: a black and white image containing the road edges
    """
    height, width, channels = img.shape
    img_output = np.zeros((height, width, 3), np.uint8)

    # cv2 uses BRG, so when using cv2 the tuple has to be reversed.
    # CARLA uses RGB, so the tuple can be as is.

    # color of lane marking (157, 234, 50)
    lower_mask = np.array([147, 224, 40])
    upper_mask = np.array([167, 244, 60])
    masked_marking = cv2.inRange(img, lower_mask, upper_mask)

    # color of the street (128, 64, 128)
    lower_mask = np.array([118, 54, 118])
    upper_mask = np.array([138, 74, 138])
    masked_street = cv2.inRange(img, lower_mask, upper_mask)

    masked_image = cv2.bitwise_or(masked_marking, masked_street)

    # find the contour with the largest area which is the street
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    street, largest_area = None, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            street = contour

    cv2.drawContours(img_output, street, -1, (255, 255, 255), 1)

    img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)

    return img_output


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor):
        """Constructor method. Spawn the sensor attached to a parent actor in the simulation. """
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=parent_actor)

        self.sensor.listen(lambda event: self.on_collision(event))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
        self.sensor = None

    def on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))


# ==============================================================================
# -- Lane Invasion Sensor ------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor):
        """Constructor method. Spawn the sensor attached to a parent actor in the simulation. """
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)

        self.sensor.listen(lambda event: self.on_invasion(event))

    def on_invasion(self, event):
        """On invasion method"""
        lane_types = set(x.type for x in event.crossed_lane_markings)
        # TODO lane types filtern/ oder nicht, wenn es nicht funktioniert

        # (lane_types)

        self.history.append(event)


# ==============================================================================
# -- Checkpoint Manager  -------------------------------------------------------
# ==============================================================================
class CheckpointManager:
    """
    Class maintaining the list of checkpoints the car has to go through. The next checkpoint the car should drive to is
    active. When a car goes through a checkpoint the next one will be selected as active.
    """

    def __init__(self):
        self.checkpoints = []
        self.current = 0

    def init_checkpoints(self):
        self.checkpoints.append(Checkpoint((56.29, 300.03), (59.0, 310.61)))  # 1
        self.checkpoints.append(Checkpoint((40.56, 290.16), (47.28, 294.21)))  # 2
        self.checkpoints.append(Checkpoint((39.88, 270.72), (47.94, 274.30)))  # 3
        self.checkpoints.append(Checkpoint((39.65, 249.20), (47.81, 253.73)))  # 4
        self.checkpoints.append(Checkpoint((54.45, 235.06), (60.21, 243.01)))  # 5
        self.checkpoints.append(Checkpoint((81.25, 234.85), (60.21, 243.01)))  # 6
        self.checkpoints.append(Checkpoint((101.19, 234.56), (105.04, 242.99)))  # 7
        self.checkpoints.append(Checkpoint((121.11, 234.95), (125.84, 242.78)))  # 8
        self.checkpoints.append(Checkpoint((130.34, 225.01), (138.09, 229.04)))  # 9
        self.checkpoints.append(Checkpoint((130.26, 211.45), (138.26, 215.02)))  # 10
        self.checkpoints.append(Checkpoint((130.03, 199.36), (138.28, 202.17)))  # 11
        self.checkpoints.append(Checkpoint((142.63, 187.11), (146.67, 193.79)))  # 12
        self.checkpoints.append(Checkpoint((164.83, 185.64), (158.93, 193.39)))  # 13
        self.checkpoints.append(Checkpoint((178.83, 185.67), (183.42, 193.82)))  # 14
        self.checkpoints.append(Checkpoint((187.23, 199.77), (193.71, 202.44)))  # 15
        self.checkpoints.append(Checkpoint((187.68, 211.07), (195.74, 214.72)))  # 16
        self.checkpoints.append(Checkpoint((187.50, 226.63), (195.69, 230.63)))  # 17
        self.checkpoints.append(Checkpoint((187.71, 248.95), (195.60, 251.40)))  # 18
        self.checkpoints.append(Checkpoint((187.79, 272.25), (195.47, 276.07)))  # 19
        self.checkpoints.append(Checkpoint((187.70, 292.79), (195.86, 296.51)))  # 20
        self.checkpoints.append(Checkpoint((177.23, 300.79), (182.63, 308.80)))  # 21
        self.checkpoints.append(Checkpoint((147.25, 300.61), (152.20, 308.55)))  # 22
        self.checkpoints.append(Checkpoint((101.82, 300.65), (106.76, 308.59)))  # 23
        self.checkpoints.append(Checkpoint((76.28, 300.46), (80.43, 308.71)))  # 0

    def reset(self):
        self.current = 0

    def check_in_current(self, position: tuple) -> bool:
        if self.current is None:
            return False

        return self.checkpoints[self.current].is_inbounds(position)

    def toggle_next(self):
        self.current += 1

    def check_finished(self):
        return self.current >= len(self.checkpoints)


# ==============================================================================
# -- Checkpoints    ------------------------------------------------------------
# ==============================================================================
class Checkpoint:
    """Class containing two points which span a checkpoints area. """

    def __init__(self, p0: tuple, p1: tuple):
        # for orientation see checkpoint overview screenshot
        self.top_left = p0
        self.bottom_right = p1

    def is_inbounds(self, pos: tuple) -> bool:
        return self.top_left[0] <= pos[0] <= self.bottom_right[0] and \
               self.top_left[1] <= pos[1] <= self.bottom_right[1]

    def get_location(self) -> carla.Location:
        """Get a location object in the middle of the checkpoint. """
        return carla.Location(x=(self.top_left[0] + self.bottom_right[0]) / 2,
                              y=(self.top_left[1] + self.bottom_right[1]) / 2)


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def learn_loop():
    global epsilon

    car_environment = None
    agent = None

    FPS = 60
    ep_rewards = [-200]

    try:
        # connect to the CARLA simulator
        client = carla.Client(PORT, HOST)
        client.set_timeout(5.0)

        client.load_world("Town02_Opt")
        # TODO layer unloaden, in dem kleine Sachen sind, die Kollisionen verursachen könnten, die wir nicht gebrauchen können.

        sim_world = client.get_world()

        # create car environment in the simulator and our Reinforcement Learning agent
        checkpoint_manager = CheckpointManager()
        car_environment = CarEnvironment(sim_world, checkpoint_manager)
        agent = DQNAgent()

        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        start_state = np.ones((1, HEIGHT, WIDTH, 1)), 1, 1, 1

        agent.get_qs(start_state)

        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = car_environment.restart()

            # Reset flag and start iterating until episode ends
            car_environment.done = False
            car_environment.episode_start = time.time()
            car_environment.extra_time = 0

            # Play for given number of seconds only
            while True:

                # get fitting action from Q table for the current state, or select one at random
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                    # print(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, 9)
                    time.sleep(1 / FPS)

                action = action % 3
                # execute the action in the environment
                new_state, reward, done, _ = car_environment.step(action)

                # Transform new continuous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            car_environment.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                               epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(
                        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            print(str(episode_reward) + " :Reward|Epsilon: " + str(epsilon))

        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    finally:
        if car_environment is not None:
            car_environment.destroy()

        # pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', HOST, PORT)

    # create models folder to save the progress in
    if not os.path.isdir('models'):
        os.makedirs('models')

    # start the learning process
    try:
        learn_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
