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
        self.camera_manager = CameraManager(self.agent.actor, debug=True)

        self.camera_manager.spawn_cameras()

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
