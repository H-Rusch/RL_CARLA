from __future__ import print_function
import os

from RL_Agent import DQNAgent, MODEL_NAME
from CheckpointManager import CheckpointManager
from Simulator import CarEnvironment
from Simulator import WIDTH, HEIGHT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import logging
import sys
import subprocess

import time
import numpy as np

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

MEMORY_FRACTION = 0.4
MIN_REWARD = 4

EPISODES = 100

epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
MIN_EPSILON_2 = 0.01

AGGREGATE_STATS_EVERY = 10

load_model_name = None


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def start_carla():
    print("startCarla")
    subprocess.Popen("..\\..\\CarlaUE4.exe -quality-level=Low -ResX=300 -ResY=200")
    time.sleep(3)

    while True:

        try:
            client = carla.Client(HOST, PORT)
            client.set_timeout(5.0)
            map_name = client.get_world().get_map().name
            # TODO layer unloaden, in dem kleine Sachen sind, die Kollisionen verursachen könnten, die wir nicht gebrauchen können.
            if map_name != "Town02_Opt":
                client.load_world("Town02_Opt")
                time.sleep(1)
            return client.get_world()
        except RuntimeError:
            time.sleep(0.1)


def learn_loop(sim_world):
    print("learn start")
    global epsilon
    global load_model_name

    car_environment = None
    agent = None
    trainer_thread = None

    FPS = 60
    ep_rewards = [-200]

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

            # Reset environment and get initial state
            current_state = car_environment.restart()

            # Reset flag and start iterating until episode ends
            car_environment.done = False
            car_environment.episode_start = time.time()
            car_environment.extra_time = 0

            # Play for given number of seconds only
            while True:
                # get fitting action from Q table for the current state, or select one at random
                if current_state[3] == 0:
                    action = 1
                elif np.random.random() > epsilon:
                    qs = agent.get_qs(current_state)
                    action = np.argmax(qs)
                else:
                    action = np.random.randint(0, 9)
                    time.sleep(1 / FPS)

                # action = action % 3
                # execute the action in the environment
                new_state, reward, done, _ = car_environment.step(action)

                # Transform new continuous state to new discrete state and count reward
                episode_reward += reward
                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            car_environment.destroy()
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
                    save_model(
                        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model',
                        agent)

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            if epsilon < MIN_EPSILON_2:
                epsilon = 0.5
            print(str(episode_reward) + " :Reward|Epsilon: " + str(epsilon))

            if episode % 1000 == 0:
                save_model(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model',
                    agent)

    finally:
        print("learn finally")
        # Set termination flag for training thread and wait for it to finish
        if agent is not None:
            agent.terminate = True
            if trainer_thread is not None:
                trainer_thread.join()
            save_model(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model',
                agent)

        if car_environment is not None:
            car_environment.destroy()


def save_model(model_name, agent):
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
    """Main method"""

    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', HOST, PORT)

    # create models folder to save the progress in
    if not os.path.isdir('models'):
        os.makedirs('models')

    # start the learning process
    while True:
        try:
            sim_world = start_carla()
            learn_loop(sim_world)
        except RuntimeError:
            print("main exception")
            time.sleep(0.1)
        except KeyboardInterrupt as e:
            print('\nCancelled by user.')
            raise e


if __name__ == '__main__':
    main()
