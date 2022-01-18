from __future__ import print_function
import os
import platform

from RL_Agent import DQNAgent, MODEL_NAME
from CheckpointManager import CheckpointManager
from Simulator import CarEnvironment
from Simulator import WIDTH, HEIGHT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import logging
import psutil
import sys
import subprocess

import time
import numpy as np

from threading import Thread

import tensorflow as tf

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

EPISODES = 100

epsilon = 1
EPSILON_DECAY = 0.985
MIN_EPSILON = 0.001
MIN_EPSILON_2 = 0.1

AGGREGATE_STATS_EVERY = 10

load_model_name = None


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
    kill_processes()
    subprocess.Popen(f"../../{EXECUTABLE} -quality-level=Low -ResX=300 -ResY=200")
    time.sleep(3)

    while True:
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
        except RuntimeError as e:
            time.sleep(0.1)

def learn_loop(sim_world):
    global epsilon
    global load_model_name

    car_environment = None
    agent = None
    trainer_thread = None

    FPS = 60
    ep_rewards = [-200]
    actions = [1]

    try:
        # create car environment in the simulator and our Reinforcement Learning agent
        checkpoint_manager = CheckpointManager()
        car_environment = CarEnvironment(sim_world, checkpoint_manager)
        agent = DQNAgent(load_model_name)

        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        if load_model_name is None:
            start_state = np.ones((1, HEIGHT, WIDTH, 1)), 1, 1, 1
            agent.get_qs(start_state)

        episode = 0
        while True:
            episode += 1
            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # reset environment and get initial state
            current_state = car_environment.restart()

            # take the time to be able to stop the episode
            car_environment.episode_start = time.time()
            car_environment.extra_time = 0

            # drive until the time runs out
            while True:
                # get fitting action from Q table for the current state, or select one at random
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

                episode_reward += reward
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if done:
                    break

            # clean up the simulator by destroying the car and the sensors
            car_environment.destroy()

            # append episode reward to a list and log stats every given number of episodes
            ep_rewards.append(episode_reward)
            if episode % AGGREGATE_STATS_EVERY == 0 or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                               epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    saveModel(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model', agent)


            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            avg_a = sum(actions) / len(actions)
            if epsilon < MIN_EPSILON_2 and not any(abs(ac-avg_a) > 1 for ac in actions):
                epsilon = 0.5
            print(str(episode_reward) + " :Reward|Epsilon: " + str(epsilon))

            if episode % 1000 == 0:
                saveModel(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model', agent)     

        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    finally:
        # Set termination flag for training thread and wait for it to finish
        if agent is not None:
            agent.terminate = True
            if trainer_thread is not None:
                trainer_thread.join()
            saveModel(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model', agent)

        if car_environment is not None:
            car_environment.destroy()

def saveModel(model_name, agent):
    print("save ", model_name)
    global load_model_name
    agent.model.save(model_name)
    load_model_name = model_name

def action_to_s(action):
    if action == 0:
        return "accelerate left"
    elif action == 1:
        return "accelerate straight"
    elif action == 2:
        return "accelerate right"
    elif action == 3:
        return "steady left"
    elif action == 4:
        return "steady straight"
    elif action == 5:
        return "steady right"
    elif action == 6:
        return "brake left"
    elif action == 7:
        return "brake straight"
    elif action == 8:
        return "brake right"

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
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # create models folder to save the progress in
    if not os.path.isdir('models'):
        os.makedirs('models')

    # start the learning process
    try:
        while True:
            try:
                sim_world = start_carla()
                learn_loop(sim_world)
            except RuntimeError as e:
                print("runtime error")
                time.sleep(0.1)
            except Exception as e:
                print("main exception")
                time.sleep(0.1)
    except KeyboardInterrupt as e:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
