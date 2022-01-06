from __future__ import print_function
import os

from RL_Agent import DQNAgent, MODEL_NAME, START_MODEL
from CheckpointManager import CheckpointManager
from Simulator import CarEnvironment
from Simulator import WIDTH, HEIGHT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# TODO testen, ob bei den Schwarz/ weiß bildern die Werte zwischen 0 und 1 sind. Wenn ja müssen wir die Bilder nicht mehr durch 255 teilen

import glob
import logging
import sys

import time
import numpy as np

from threading import Thread

from tqdm import tqdm

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

PORT = "127.0.0.1"
HOST = 2000



MEMORY_FRACTION = 0.4
MIN_REWARD = -1

EPISODES = 100

epsilon = 1
EPSILON_DECAY = 0.95  ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def learn_loop():
    global epsilon

    car_environment = None
    agent = None

    FPS = 60
    ep_rewards = [-200]

    try:
        # connect to the CARLA simulator
        client = carla.Client(PORT, HOST)
        client.set_timeout(5.0)

        client.load_world("Town02_Opt")
        # TODO layer unloaden, in dem kleine Sachen sind, die Kollisionen verursachen könnten, die wir nicht gebrauchen können.

        sim_world = client.get_world()

        # create car environment in the simulator and our Reinforcement Learning agent
        checkpoint_manager = CheckpointManager()
        car_environment = CarEnvironment(sim_world, checkpoint_manager)
        agent = DQNAgent()

        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        if START_MODEL is None:
            start_state = np.ones((1, HEIGHT, WIDTH, 1)), 1, 1, 1
            agent.get_qs(start_state)

        episode = 0

        while True:
        #for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
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
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                    # print(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, 9)
                    time.sleep(1 / FPS)

                action = action % 3
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
                    agent.model.save(
                        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            print(str(episode_reward) + " :Reward|Epsilon: " + str(epsilon))

            if episode % 1000 == 0:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    finally:
        if car_environment is not None:
            car_environment.destroy()

        # pygame.quit()


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
    try:
        learn_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user.')


if __name__ == '__main__':
    main()
