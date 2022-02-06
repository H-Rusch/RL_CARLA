from __future__ import print_function
import os
import platform

from collections import deque
from ModifiedTensorboard import ModifiedTensorBoard
from RL_Agent import DQNAgent, MODEL_NAME, REPLAY_MEMORY_SIZE
from CheckpointManager import CheckpointManager
from Simulator import CarEnvironment
from Simulator import WIDTH, HEIGHT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import logging
import psutil
import sys
import traceback
import subprocess

import cv2
import tensorflow as tf
import time
import numpy as np

import threading
from threading import Thread

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

MEMORY_FRACTION = 0.4
MIN_REWARD = 4

episode = 0

epsilon = 1
EPSILON_DECAY = 0.985
MIN_EPSILON = 0.001
MIN_EPSILON_2 = 0.1

AGGREGATE_STATS_EVERY = 10
SAVE_MODEL_EVERY = 1000

# model to which should be loaded. None to create a new model
load_model_name = None

FPS = 60


# ==============================================================================
# -- Learning Algorithm  -------------------------------------------------------
# ==============================================================================

def start_carla() -> carla.World:
    """
    Start the CARLA simulator. Loads the correct map and unloads map layers which might cause problems.
    :return: the world object in the CARLA simulator
    """
    while True:
        log_info(time.strftime("%H %M") + ": Carla Start\n")
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
            return client.get_world()
        except RuntimeError:
            time.sleep(0.1)


def kill_processes():
    """ Stop running CARLA processes in order to start a new one. """
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

    # kill process and wait until it's being killed
    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)


def learn_loop(sim_world, tensorboard, replay_memory):
    global episode
    global epsilon
    global load_model_name

    car_environment = None
    agent = None
    trainer_thread = None

    ep_rewards = [-200]

    try:
        # create car environment in the simulator and our Reinforcement Learning agent
        checkpoint_manager = CheckpointManager()
        car_environment = CarEnvironment(sim_world, checkpoint_manager)
        agent = DQNAgent(load_model_name, tensorboard, replay_memory)

        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        if load_model_name is None:
            start_state = np.ones((1, HEIGHT, WIDTH, 1)) * 255, 1, 1, 1
            agent.get_qs(start_state)

        while True:
            episode += 1
            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # drive until the time runs out
            actions, episode_reward = execute_episode(agent, car_environment)

            # append episode reward to a list and log stats every given number of episodes
            ep_rewards.append(episode_reward)
            if episode % AGGREGATE_STATS_EVERY == 0 or episode == 1:
                average_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                               epsilon=epsilon)

            # decay epsilon each iteration
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            avg_a = sum(actions) / len(actions)
            # reset epsilon
            if epsilon < MIN_EPSILON_2 and not any(abs(ac - avg_a) > 1 for ac in actions):
                epsilon = 0.5

            print(str(episode_reward) + " :Reward|Epsilon: " + str(epsilon))

            if episode % SAVE_MODEL_EVERY == 0:
                average_reward = np.mean(ep_rewards[-SAVE_MODEL_EVERY:])
                min_reward = min(ep_rewards[-SAVE_MODEL_EVERY:])
                max_reward = max(ep_rewards[-SAVE_MODEL_EVERY:])

                save_model(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model',
                    agent)
                log_info(time.strftime("%H %M") + ": Save episodes\n")

            if threading.active_count() < 2:
                raise RuntimeError()

        # Set termination flag for training thread and wait for it to finish

    finally:
        # Set termination flag for training thread and wait for it to finish
        if agent is not None:
            agent.terminate = True
            if trainer_thread is not None:
                trainer_thread.join()

            save_model(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model',
                agent)
            log_info(time.strftime("%H %M") + ": Save finally\n")

        if car_environment is not None:
            car_environment.destroy()


def execute_episode(agent: DQNAgent, car_environment: CarEnvironment) -> tuple:
    """
    Execute a full episode.
    First reset the world in the simulator and reset the episode time.
    Continuously take driving actions until the time runs out or the car gets into a collision.
    How an action is chosen is based on the epsilon value.
    If a generated value is lower than the epsilon value, a random action will be selected, but if the generated value
    is higher, the value with the best q values for the current state will be selected.

    :param agent: the agent doing the learning
    :param car_environment: the car environment containing the car and all sensors
    :return: tuple containing a list of the actions taken this episode and the summed reward in this episode
    """
    # reset environment and get initial state
    current_state = car_environment.restart()

    actions = [1]
    episode_reward = 0
    standing = True

    car_environment.episode_start = time.time()
    car_environment.extra_time = 0

    while True:
        if standing and current_state[3] != 0:
            standing = False

        # if the car is standing still go forwards
        if current_state[3] == 0:
            action = 1
            time.sleep(12 / FPS)
        elif np.random.random() > epsilon:
            qs = agent.get_qs(current_state)
            action = np.argmax(qs)
            if len(actions) > 200:
                actions.pop(0)
            actions.append(action)
        else:
            action = np.random.randint(0, 9)
            time.sleep(12 / FPS)

        # execute the action in the environment
        new_state, reward, done, _ = car_environment.step(action)

        if not standing:
            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))

        current_state = new_state

        if done:
            print(actions)
            break

    # clean up the simulator by destroying the car and the sensors
    car_environment.destroy()

    return actions, episode_reward


def save_model(model_name, agent):
    print("save ", model_name)
    global load_model_name
    agent.model.save(model_name)
    load_model_name = model_name


def log_info(info: str):
    with open("log.txt", "a") as file:
        file.write(info)


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    """
    Keep restarting the simulator and the learning process when the simulator crashes.
    """
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', HOST, PORT)
    print("Number of GPUs available: ", len(tf.config.list_physical_devices('GPU')))

    # create models folder to save the progress in
    if not os.path.isdir('models'):
        os.makedirs('models')

    tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    # start the learning process
    try:
        while True:
            try:
                sim_world = start_carla()
                learn_loop(sim_world, tensorboard, replay_memory)
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
