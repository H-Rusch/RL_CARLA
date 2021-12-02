from __future__ import print_function

import glob
import logging
import math
import os
import numpy.random as random
import sys
import weakref

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


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

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

        self.agent = CustomAgent(self)
        self.collision_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None

        self.restart()

    def restart(self):
        """Restart the world"""
        # clean up old objects
        self.destroy()

        # spawn the actor
        self.agent.spawn_actor()

        self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.agent.actor)
        self.gnss_sensor = GnssSensor(self.agent.actor)
        self.camera_manager = CameraManager(self.agent.actor)
        self.camera_manager.transform_index = 0

        # right now 0 is rgb camera, 2 is semantic segmentation
        self.camera_manager.set_sensor(2)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager,
            self.gnss_sensor,
            self.collision_sensor,
            self.agent]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- AI-Agent ------------------------------------------------------------------
# ==============================================================================

class CustomAgent:
    def __init__(self, world: World):
        # the car actually driving around in the simulation
        self.actor = None
        self.world = world

        # the model of the car driving around
        self.blueprint = random.choice(self.world.world.get_blueprint_library().filter("model3"))
        self.blueprint.set_attribute('role_name', 'hero')

        # spawn point of the car
        spawn_points = self.world.map.get_spawn_points()
        if spawn_points is None:
            print('There are no spawn points available in your map/town.')
            print('Please add some Vehicle Spawn Point to your UE4 scene.')
            sys.exit(1)
        self.spawn_point = random.choice(spawn_points)

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
        self = weak_self()
        if not self:
            return
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

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        attachment = carla.AttachmentType

        # transform for camera. POV of the vehicle
        self._camera_transform = (carla.Transform(
            carla.Location(x=2, z=1)), attachment.Rigid)

        # sensors for this camera manager
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)']]

        # get the blueprints for the sensors
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(WIDTH))
                blp.set_attribute('image_size_y', str(HEIGHT))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
                blp.set_attribute('channels', '50')
                blp.set_attribute('rotation_frequency', '10')
            item.append(blp)

        self.index = None

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
        self.sensor = None
        self.index = None

    def set_sensor(self, index, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
                force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None

            # spawn the camera and attach it to the parent vehicle
            self.sensor = self._parent.get_world() \
                .spawn_actor(self.sensors[index][-1],
                             self._camera_transform[0],
                             attach_to=self._parent,
                             attachment_type=self._camera_transform[-1])

            # We need to pass the lambda a weak reference to self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

        self.index = index

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):

            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min((WIDTH, HEIGHT)) / 100.0
            lidar_data += (0.5 * WIDTH, 0.5 * HEIGHT)
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (WIDTH, HEIGHT, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img = np.reshape(img, (image.height, image.width, 4))
            img = img[:, :, :3]
            img = img[:, :, ::-1]

            height, width, channels = img.shape
            img_output = np.zeros((height, width, 3), np.uint8)

            # create masked black and white image where only the street is visible

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

            self.surface = pygame.surfarray.make_surface(img_output.swapaxes(0, 1))


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop():
    """
    Main loop of the simulation.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(PORT, HOST)
        client.set_timeout(5.0)

        sim_world = client.get_world()

        display = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

        world = World(sim_world)

        clock = pygame.time.Clock()

        running = True
        while running:
            # handle key events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        world.restart()

            if not running:
                break

            clock.tick()
            world.world.wait_for_tick()

            world.render(display)
            pygame.display.flip()

    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """Main method"""

    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', HOST, PORT)

    try:
        game_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
