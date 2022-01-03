#!/usr/bin/env python

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

_HOST_ = '127.0.0.1'
_PORT_ = 2000
_SLEEP_TIME_ = 3


def main():
    client = carla.Client(_HOST_, _PORT_)
    client.set_timeout(2.0)

    client.load_world("Town02_Opt")
    world = client.get_world()
    map = world.get_map()

    while True:
        t = world.get_spectator().get_transform()
        x, y, z = t.location.x, t.location.y, t.location.z
        xa, ya = str(x)[:str(x).find(".") + 3], str(y)[:str(y).find(".") + 3]

        coordinate_str = f"{xa},\t{ya}"
        wp = map.get_waypoint(carla.Location(x=x, y=y, z=z))

        world.debug.draw_string(wp.transform.location, 'O', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=5.0,
                                persistent_lines=True)

        print(coordinate_str)
        time.sleep(_SLEEP_TIME_)


if __name__ == '__main__':
    main()
