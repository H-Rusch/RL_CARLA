import os
import glob
import sys

import time
import math
import random

import cv2
import numpy as np

from CheckpointManager import CheckpointManager

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

WIDTH = 640
HEIGHT = 480

TARGET_SPEED = 35
DEGREE_DIVISOR = 360

SHOW_IMAGE = False
SECONDS_PER_EPISODE = 10
SPAWN_LOCATION = (79.19, 302.39, 2.0)
FOV = 110


# ==============================================================================
# -- CarEnvironment ------------------------------------------------------------
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
        :returns: The current state the environment. See 'Simulation.get_state()'
        """

        # clean up old objects
        self.destroy()

        self.checkpoint_manager.reset()

        # spawn the actor
        self.vehicle.spawn_actor()

        # set up and spawn the sensors
        self.camera_manager = CameraManager(self.vehicle.actor, show_image=SHOW_IMAGE)
        self.collision_sensor = CollisionSensor(self.vehicle.actor)

        self.camera_manager.spawn_cameras()

        # wait until images are recorded
        while self.camera_manager.lane_detection_img is None:
            time.sleep(0.01)

        time.sleep(1)

        self.episode_start = time.time()

        return self.get_state()

    def step(self, action: int) -> tuple:
        """
        Execute a step in the car environment. The actor will be rewarded based on the action he took.
        If the actor manages to drive through a checkpoint, the next one will be toggled and the maximum time of the
        current episode will be incremented.

        :param action: the action the car should perform, selected by the neural network
        :return: a tuple of (current state the actor is in; the reward earned in this step; an information whether the
        episode is over; No additional information)
        """
        reward = 0
        done = False

        self.vehicle.execute_action(action)
        car_location = (self.vehicle.actor.get_location().x, self.vehicle.actor.get_location().y)
        if self.checkpoint_manager.check_in_current(car_location):
            reward += 5

            # give extra reward based on how much time was left when reaching the checkpoint
            time_left = (self.episode_start + SECONDS_PER_EPISODE + self.extra_time) - time.time()
            if time_left > 0:
                reward += time_left / 5
            self.checkpoint_manager.toggle_next()

            self.extra_time += 10
            self.checkpoint_distance = 10_000

            # Show next checkpoint:
            # self.world.debug.draw_point(
            #    self.checkpoint_manager.checkpoints[self.checkpoint_manager.current].get_location(),
            #    size=1.0,
            #    color=carla.Color(r=255, g=0, b=0),
            #    life_time=5.0)

            if self.checkpoint_manager.check_finished():
                reward += 100
                done = True

        kmh = self.vehicle.get_kmh()
        if not done:

            if len(self.collision_sensor.history) != 0:
                done = True
                reward -= 10
            elif kmh < 5:
                done = False
                reward -= 0.5
            elif kmh < TARGET_SPEED:
                done = False
                reward -= 0.01
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
            reward += 0.5
        elif distance > self.checkpoint_distance:
            reward -= 0.1

        return current_state, reward, done, None

    def get_state(self) -> tuple:
        """
        Get the current state of the actor in the simulation.

        :return: tuple containing the preprocessed camera image, the direction the next checkpoint is in, the distance
        to the next checkpoint and the vehicles speed
        """

        # current camera image
        img = self.camera_manager.lane_detection_img

        # direction and distance to the next checkpoint
        distance, direction = self.get_next_checkpoint_state()

        kmh = self.vehicle.get_kmh()

        return [img], direction, distance, kmh

    def get_next_checkpoint_state(self) -> tuple:
        """
        Calculate the distance and angle to the next active checkpoint. The angle should have 180 as the mid point, so
        a value lower than 180 means the checkpoint is on the left, and a value greater than 180 means the checkpoint
        is on the right.

        :return: tuple containing the distance and the angle to the next active checkpoint
        """
        vehicle_transform = self.vehicle.actor.get_transform()

        checkpoint = self.checkpoint_manager.checkpoints[self.checkpoint_manager.current]
        checkpoint_location = checkpoint.get_location()

        distance = int(vehicle_transform.location.distance(checkpoint_location))

        # calculate the angle between the car and the checkpoint by computing the atan2 and
        # normalizing the angle to a value in [0, 360)
        # 1 is one degree to the right, 359 is one degree to the left
        c_x, c_y = checkpoint_location.x, checkpoint_location.y
        v_x, v_y = vehicle_transform.location.x, vehicle_transform.location.y

        raw_angle = math.atan2(c_y - v_y, c_x - v_x)
        raw_angle = math.degrees(raw_angle)

        direction = int((raw_angle - vehicle_transform.rotation.yaw) % DEGREE_DIVISOR)

        # set 180 as the value for going straight
        # 179 is one degree to the left, 181 is one degree to the right
        if direction > 180:
            direction -= 180
        elif direction < 180:
            direction += 180

        return distance, direction

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
# -- Vehicle -------------------------------------------------------------------
# ==============================================================================

class Vehicle:
    """Class representing the vehicle in the simulation """

    def __init__(self, world: CarEnvironment):
        self.actor = None
        self.world = world

        # get the model for a car
        self.blueprint = random.choice(self.world.world.get_blueprint_library().filter("model3"))
        self.blueprint.set_attribute('role_name', 'hero')

        # define fixed spawn point for a car
        x, y, z = SPAWN_LOCATION
        self.spawn_point = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=180))

    def spawn_actor(self):
        self.actor = self.world.world.spawn_actor(self.blueprint, self.spawn_point)

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
    """ Class for camera management """

    def __init__(self, parent_actor, show_image: bool = False):
        """
        Constructor Method
        :param parent_actor: the actor the cameras are attached to
        :param show_image: information whether the image the camera records should be shown
        """
        self.rgb_camera = None
        self.sem_seg_camera = None

        self.lane_detection_img = None

        self.show = show_image
        self._parent = parent_actor

        # transforms for camera. A first-person-view and a third-person-view are defined
        self._camera_transform_fp = (carla.Transform(
            carla.Location(x=2.2, z=1)), carla.AttachmentType.Rigid)
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
                             self._camera_transform_fp[0],
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("", img)
        cv2.waitKey(0)

    def parse_image(self, image):
        """Parse the semantic segmentation image into the lane detection image."""
        image.convert(self.sensors[1][1])
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]

        self.lane_detection_img = lane_detection_from_sem_seg(img)


def lane_detection_from_sem_seg(img):
    """
    Convert a semantic segmentation image into a black and white image where only road is highlighted.
    :param img: the image returned form the semantic segmentation camera
    :return: a black and white image containing the road's surface in white
    """

    # cv2 uses BRG, so when using cv2 the tuple has to be reversed.
    # CARLA uses RGB, so the tuple can be as is.

    # color of lane marking (157, 234, 50)
    lower_mask = np.array([145, 190, 40])
    upper_mask = np.array([167, 255, 80])
    masked_marking = cv2.inRange(img, lower_mask, upper_mask)

    # color of the street (128, 64, 128)
    lower_mask = np.array([118, 54, 118])
    upper_mask = np.array([138, 74, 138])
    masked_street = cv2.inRange(img, lower_mask, upper_mask)

    masked_image = cv2.bitwise_or(masked_marking, masked_street)

    return masked_image


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    """ Class for collision sensors """

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
        """
        Append a collision to the collision history.
        """
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
