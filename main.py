from __future__ import print_function

import glob
import logging
import math
import os
import numpy.random as numpy_random
import sys
import weakref

import random
import time
import numpy as np
import cv2
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizer_v2.adam import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

import tensorflow as tf
from threading import Thread

from tqdm import tqdm

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

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

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
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

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
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
# -- CarEnvironment ---------------------------------------------------------------
# ==============================================================================

class CarEnvironment(object):
    """ Class representing the surrounding environment """
    STEER_AMT = 1.0

    def __init__(self, carla_world):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        self.vehicle = Vehicle(self)
        self.collision_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None

        self.restart()

    def restart(self):
        """Restart the world"""
        # clean up old objects
        self.destroy()

        # spawn the actor
        self.vehicle.spawn_actor()

        self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle.actor)
        self.gnss_sensor = GnssSensor(self.vehicle.actor)
        self.camera_manager = CameraManager(self.vehicle.actor, debug=False)

        self.camera_manager.spawn_cameras()

        while self.camera_manager.lane_detection_img is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        return self.camera_manager.lane_detection_img

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)

    def step(self, action):
        if action == 0:
            self.vehicle.actor.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.actor.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.actor.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.actor.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_sensor.history) != 0:
            done = True
            reward = -1
        elif kmh < 50:
            done = False
            reward = -0.1
        else:
            done = False
            reward = 0.1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.camera_manager.lane_detection_img, reward, done, None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager,
            self.gnss_sensor,
            self.collision_sensor,
            self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- Vehicle ------------------------------------------------------------------
# ==============================================================================

class Vehicle:
    def __init__(self, world: CarEnvironment):
        # the car actually driving around in the simulation
        self.actor = None
        self.world = world

        # the model of the car driving around
        self.blueprint = numpy_random.choice(self.world.world.get_blueprint_library().filter("model3"))
        self.blueprint.set_attribute('role_name', 'hero')

        # spawn point of the car
        spawn_points = self.world.map.get_spawn_points()
        if spawn_points is None:
            print('There are no spawn points available in your map/town.')
            print('Please add some Vehicle Spawn Point to your UE4 scene.')
            sys.exit(1)
        self.spawn_point = numpy_random.choice(spawn_points)

    def spawn_actor(self):
        self.actor = self.world.world.try_spawn_actor(self.blueprint, self.spawn_point)

        self.modify_vehicle_physics()
        # self.actor.set_autopilot(True)

        self.actor.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    def modify_vehicle_physics(self):
        try:
            physics_control = self.actor.get_physics_control()
            physics_control.use_gear_autobox(True)
            # physics_control.use_sweep_wheel_collision = True

            self.actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def destroy(self):
        if self.actor is not None:
            self.actor.destroy()

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
        #self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        #with self.graph.as_default():
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        #with self.graph.as_default():
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

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        #with self.graph.as_default():
        self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        #with self.graph.as_default():
        self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform())
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, debug: bool = False):
        """
        Constructor Method
        :param parent_actor: the actor the cameras are attached to
        :param debug: information whether the camera manager should go into debug mode, showing multiple cameras at a time
        """
        self.rgb_camera = None
        self.sem_seg_camera = None
        self.rgb_surface = None
        self.sem_seg_surface = None
        self.lane_detection_surface = None

        self.debug = debug
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
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(WIDTH))
                bp.set_attribute('image_size_y', str(HEIGHT))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
                bp.set_attribute('channels', '50')
                bp.set_attribute('rotation_frequency', '10')
            item.append(bp)

    def destroy(self):
        """Destroy the sensors in the simulation."""
        if self.sem_seg_camera is not None:
            self.sem_seg_camera.destroy()
        self.sem_seg_camera = None

        if self.debug:
            if self.rgb_camera is not None:
                self.rgb_camera.destroy()
            self.rgb_camera = None

    def spawn_cameras(self):
        """
        Spawn the cameras in the simulation.
        Spawns only the semantic segmentation camera in normal mode and an additional rgb camera in debug mode
        """
        if self.sem_seg_camera is None:
            self.destroy()
            self.rgb_surface = None
            self.sem_seg_surface = None
            self.lane_detection_surface = None

        # We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)

        if self.debug:
            self.rgb_camera = self._parent.get_world() \
                .spawn_actor(self.sensors[0][-1],
                             self._camera_transform_tp[0],
                             attach_to=self._parent,
                             attachment_type=self._camera_transform_tp[-1])
            self.rgb_camera.listen(lambda image: CameraManager._parse_image(weak_self, image, 0))

        self.sem_seg_camera = self._parent.get_world() \
            .spawn_actor(self.sensors[1][-1],
                         self._camera_transform_fp[0],
                         attach_to=self._parent,
                         attachment_type=self._camera_transform_fp[-1])
        self.sem_seg_camera.listen(lambda image: CameraManager._parse_image(weak_self, image, 1))

    def render(self, display):
        """Render camera images.
        debug mode:
            Top Left: RGB image
            Top Right: Semantic Segmentation image
            Bottom Left: Lane Edges
        normal mode:
            only lane Edges
        """
        if self.debug:
            if all([surface is not None for surface in
                    [self.rgb_surface, self.sem_seg_surface, self.lane_detection_surface]]):
                display.blit(self.rgb_surface, (0, 0))
                display.blit(self.sem_seg_surface, (WIDTH / 2, 0))
                display.blit(self.lane_detection_surface, (0, HEIGHT / 2))

        else:
            if self.lane_detection_surface is not None:
                display.blit(self.lane_detection_surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, number):
        """Set the captured image of the selected camera as the surface to be displayed."""
        self = weak_self()
        if not self:
            return

        image.convert(self.sensors[number][1])
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]

        if self.debug:
            # scale down image in debug mode to show multiple at the same time
            d_size = (int(WIDTH / 2), int(HEIGHT / 2))
            img = cv2.resize(img, d_size)

            if number == 0:
                self.rgb_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            elif number == 1:
                self.sem_seg_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                img = lane_detection_from_sem_seg(img)
                self.lane_detection_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

        else:
            img = lane_detection_from_sem_seg(img)
            self.lane_detection_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))


def lane_detection_from_sem_seg(img):
    """
    Convert a semantic segmentation image into a black and white image where only the contours of the road are highlighted.
    :param img: the image returned form the semantic segmentation camera
    :return: a black and shite image containing the road edges
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
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop():
    """
    Main loop of the simulation.
    """
    global epsilon

    #pygame.init()
    #pygame.font.init()
    carEnv = None

    FPS = 60
    # For stats
    ep_rewards = [-200]

    try:
        client = carla.Client(PORT, HOST)
        client.set_timeout(5.0)

        sim_world = client.get_world()

        display = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

        carEnv = CarEnvironment(sim_world)
        agent = DQNAgent()

        clock = pygame.time.Clock()

        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        agent.get_qs(np.ones((HEIGHT, WIDTH, 3)))

        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

            # carEnv.collision_sensor.history = []

            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = carEnv.restart()

            # Reset flag and start iterating until episode ends
            carEnv.done = False
            carEnv.episode_start = time.time()

            '''running = True
            while running:
                # handle key events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            carEnv.restart()

                if not running:
                    break

                clock.tick()
                carEnv.world.wait_for_tick()

                carEnv.render(display)
                pygame.display.flip()'''

            # Play for given number of seconds only
            while True:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, 3)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1 / FPS)

                new_state, reward, done, _ = carEnv.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            carEnv.destroy()

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

        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    finally:
        if carEnv is not None:
            carEnv.destroy()

        #pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', HOST, PORT)

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    try:
        game_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
