import os

import glob
import sys

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
