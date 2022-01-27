from __future__ import print_function
import os
import platform

from collections import deque
from ModifiedTensorboard import ModifiedTensorBoard
from RL_Agent import DQNAgent, MODEL_NAME, REPLAY_MEMORY_SIZE
from CheckpointManager import CheckpointManager
from Simulator import CarEnvironment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import logging
import psutil
import sys
import traceback
import subprocess
import pygame
from pygame.locals import K_SPACE

import time
import numpy as np


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
# -- Defining Constants --------------------------------------------------------
# ==============================================================================

PORT = 2000
HOST = "127.0.0.1"

EXECUTABLE = "CarlaUE4.exe" if platform.system() == "Windows" else "CarlaUE4.sh"

load_model_name = "models/CNN____90.41max___56.48avg___34.62min__1642868073.model"


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def kill_processes():
    for process in psutil.process_iter():
        if process.name().lower().startswith(EXECUTABLE.split('.')[0].lower()):
            try:
                process.terminate()
            except:
                pass
    still_alive = []
    for process in psutil.process_iter():
        if process.name().lower().startswith(EXECUTABLE.split('.')[0].lower()):
            still_alive.append(process)

    # Kill process and wait until it's being killed
    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)


def start_carla():
    while True:
        with open("log.txt", "a") as myfile:
            myfile.write(time.strftime("%H %M") + ": Carla Start\n")
        kill_processes()
        subprocess.Popen(f"../../{EXECUTABLE} -quality-level=Low -ResX=300 -ResY=200")
        time.sleep(6)
        try:
            client = carla.Client(HOST, PORT)
            client.set_timeout(5.0)
            map_name = client.get_world().get_map().name
            if map_name != "Town02_Opt":
                client.load_world("Town02_Opt")
                time.sleep(1)

                client.get_world().unload_map_layer(carla.MapLayer.Foliage)
                time.sleep(1)

                client.get_world().unload_map_layer(carla.MapLayer.Props)
                time.sleep(1)

                client.get_world().unload_map_layer(carla.MapLayer.StreetLights)
                time.sleep(1)
            return client.get_world()
        except RuntimeError as e:
            time.sleep(0.1)


def run_loop(sim_world, tensorboard, replay_memory):
    global load_model_name

    car_environment = None

    try:
        # create car environment in the simulator and our Reinforcement Learning agent
        checkpoint_manager = CheckpointManager()
        car_environment = CarEnvironment(sim_world, checkpoint_manager)
        agent = DQNAgent(load_model_name, tensorboard, replay_memory)

        while True:
            standing = True

            # reset environment and get initial state
            current_state = car_environment.restart()

            # take the time to be able to stop the episode
            car_environment.episode_start = time.time()
            car_environment.extra_time = 0

            while True:
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.KEYUP:
                        print("TEst KEYUP", event.key)
                        if event.key == K_SPACE:
                            print("TEst K_BACKSPACE")
                            current_state = car_environment.restart()

                if standing and current_state[3] != 0:
                    standing = False

                # get fitting action from Q table for the current state
                qs = agent.get_qs(current_state)
                action = int(np.argmax(qs))

                # execute the action in the environment
                new_state, reward, done, _ = car_environment.step(action)

                current_state = new_state

    finally:
        if car_environment is not None:
            car_environment.destroy()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """
    Keep restarting the simulator and the learning process when the simulator crashes.
    """
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    screen.fill((234, 212, 252))
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', HOST, PORT)

    # create models folder to save the progress in
    if not os.path.isdir('models'):
        os.makedirs('models')

    tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    try:
        while True:
            try:
                sim_world = start_carla()
                run_loop(sim_world, tensorboard, replay_memory)
            except RuntimeError as e:
                print("runtime error", e)
                print(traceback.format_exc())
                time.sleep(0.1)
            except Exception as e:
                print("main exception", e)
                print(traceback.format_exc())
                time.sleep(0.1)
    except KeyboardInterrupt as e:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
