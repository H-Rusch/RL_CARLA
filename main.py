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
        log_info("Carla Start")
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
            except Exception:
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
            except Exception:
                pass
        psutil.wait_procs(still_alive)


def learn_loop(sim_world, tensorboard, replay_memory):
    """
    Start the learning process by creating all needed components and then continuously executing episodes.
    A given model will be loaded if the 'load_model_name' is specified. An agent is created with the model and a thread
    is started, which constantly executes the agents training. The main thread continues to executes episodes.
    When an error occurs, the current model is saved, so it can be loaded again.
    """
    global load_model_name

    car_environment = None
    agent = None
    trainer_thread = None

    episode_rewards = [-20]
    actions = deque([1], maxlen=200)

    try:
        # create car environment in the simulator and the Reinforcement Learning agent
        checkpoint_manager = CheckpointManager()
        car_environment = CarEnvironment(sim_world, checkpoint_manager)
        agent = DQNAgent(load_model_name, tensorboard, replay_memory)

        # start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        if load_model_name is None:
            start_state = np.ones((1, HEIGHT, WIDTH, 1)) * 255, 1, 1, 1
            agent.get_qs(start_state)

        # execute episodes until the program is stopped
        while True:
            execute_episode(agent, car_environment, actions, episode_rewards)

    except (RuntimeError, KeyboardInterrupt):
        avg_reward, min_reward, max_reward = calculate_rewards(episode_rewards)

        save_model(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model',
            agent)
        log_info("Save except")

    finally:
        # Set termination flag for training thread and wait for it to finish
        if agent is not None:
            agent.terminate = True
            if trainer_thread is not None:
                trainer_thread.join()

        if car_environment is not None:
            car_environment.destroy()


def execute_episode(agent: DQNAgent, car_environment: CarEnvironment, actions: deque, episode_rewards: list):
    """ Execute a full episode and do all administrative tasks regarding this episode. """
    global episode, epsilon

    episode += 1
    agent.tensorboard.step = episode

    # driving actions this episode
    episode_actions, episode_reward = execute_episode_actions(agent, car_environment)
    actions.extend(episode_actions)

    # append episode reward to a list and log stats every given number of episodes
    episode_rewards.append(episode_reward)
    if episode % AGGREGATE_STATS_EVERY == 0 or episode == 1:
        avg_reward, min_reward, max_reward = calculate_rewards(episode_rewards)
        agent.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, reward_max=max_reward,
                                       epsilon=epsilon)

    # decay epsilon each iteration
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    # reset epsilon
    if epsilon < MIN_EPSILON_2 and len(list(set(actions))) == 1:
        epsilon = 0.5

    print(f"{episode_reward} :Reward | Epsilon: {epsilon}")

    if episode % SAVE_MODEL_EVERY == 0:
        avg_reward, min_reward, max_reward = calculate_rewards(episode_rewards)

        save_model(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model',
            agent)
        log_info("Save episodes")

    if threading.active_count() < 2:
        raise RuntimeError("A Thread stopped running")


def execute_episode_actions(agent: DQNAgent, car_environment: CarEnvironment) -> tuple:
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

    episode_actions = deque(maxlen=200)
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
            episode_actions.append(action)
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
            break

    # clean up the simulator by destroying the car and the sensors
    car_environment.destroy()

    return episode_actions, episode_reward


def save_model(model_name, agent):
    global load_model_name

    print("save ", model_name)
    agent.model.save(model_name)
    load_model_name = model_name


def log_info(message: str):
    with open("log.txt", "a") as file:
        file.write(time.strftime("%H %M") + message + "\n")


def calculate_rewards(episode_rewards) -> tuple:
    avg_reward = np.mean(episode_rewards[-AGGREGATE_STATS_EVERY:])
    min_reward = min(episode_rewards[-AGGREGATE_STATS_EVERY:])
    max_reward = max(episode_rewards[-AGGREGATE_STATS_EVERY:])

    return avg_reward, min_reward, max_reward


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
    except KeyboardInterrupt:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
