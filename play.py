#!/usr/bin/env python3.6

""" Front-end script for replaying the Snake agent's behavior on a batch of episodes. """

import json
import sys
import numpy as np

from snakeai.gameplay.environment import Environment
from snakeai.gui import PyGameGUI
from snakeai.utils.cli import HelpOnFailArgumentParser

from train import create_vin_model

from vin_keras.vin import vin_model
from keras.models import Sequential
from keras.layers import *
from keras.layers.merge import *
from keras.optimizers import *
import keras.backend as K
from matplotlib import pyplot as plt
import time


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI replay client.',
        epilog='Example: play.py --agent dqn --model dqn-final.model --level 10x10.json'
    )

    parser.add_argument(
        '--interface',
        type=str,
        choices=['cli', 'gui'],
        default='gui',
        help='Interface mode (command-line or GUI).',
    )
    parser.add_argument(
        '--agent',
        required=True,
        type=str,
        choices=['human', 'dqn', 'random'],
        help='Player agent to use.',
    )
    parser.add_argument(
        '--model',
        type=str,
        help='File containing a pre-trained agent model.',
    )
    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='The number of episodes to run consecutively.',
    )
    parser.add_argument(
        '--attention',
        type=int,
        default=1,
        help='Whether to use attention',
    )

    return parser.parse_args(args)


def visualize_vin_model(env, num_last_frames, attention=1):

    image_shape = (num_last_frames, ) + env.observation_shape
    print(image_shape)

    vin = vin_model(l_s=image_shape[2], 
                    k = 20, 
                    l_q=3,
                    l_a=3,
                    attention=attention)
    vin.summary()

    q_layers = vin.layers[-3]
    visualize_q = K.function([vin.layers[0].input], [q_layers.output])


    model = Sequential()
    # Convolutions.
    model.add(Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first',
        input_shape=(num_last_frames, ) + env.observation_shape
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first'
    ))
    model.add(Activation('relu'))

    # Dense layers.
    model.add(Flatten())
    #model.add(Dense(256))
    #model.add(Activation('relu'))
    model.summary()

    merged_model = Sequential()
    merged_model.add(Merge([model, vin], mode='concat'))
    merged_model.add(Dense(256))
    merged_model.add(Activation('relu'))
    merged_model.add(Dense(env.num_actions))

    merged_model.summary()
    merged_model.compile(RMSprop(), 'MSE')

    return merged_model, visualize_q


def visualize_cli(env, agent, num_episodes=10, attention=1, visualize=None):
    """
    Play a set of episodes using the specified Snake agent.
    Use the non-interactive command-line interface and print the summary statistics afterwards.
    
    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
    """

    fruit_stats = []

    print()
    print('Playing:')
    count = 0

    for episode in range(num_episodes):
        timestep, position = env.new_episode()
        agent.begin_episode()
        game_over = False

        while not game_over:
            vis, action = agent.visualize(timestep.observation, position, timestep.reward, attention, visualize)
            #print(vis[0])
            v = np.amax(vis[0][0], axis=0)
            if count == 0 and env.stats.fruits_eaten >= 10:
                print(timestep.observation)
                print(v)
                plt.imshow(v.reshape(10,10), interpolation='nearest')
                plt.savefig('test.png')
                count += 1

            env.choose_action(action)
            timestep, position = env.timestep()
            game_over = timestep.is_episode_end

        fruit_stats.append(env.stats.fruits_eaten)

        summary = 'Episode {:3d} / {:3d} | Timesteps {:4d} | Fruits {:2d}'
        print(summary.format(episode + 1, num_episodes, env.stats.timesteps_survived, env.stats.fruits_eaten))

    print()
    print('Fruits eaten {:.1f} +/- stddev {:.1f}'.format(np.mean(fruit_stats), np.std(fruit_stats)))



def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def load_model(filename):
    """ Load a pre-trained agent model. """

    from keras.models import load_model
    return load_model(filename)


def create_agent(name, model, attention):
    """
    Create a specific type of Snake AI agent.
    
    Args:
        name (str): key identifying the agent type.
        model: (optional) a pre-trained model required by certain agents.

    Returns:
        An instance of Snake agent.
    """

    from snakeai.agent import DeepQNetworkAgent, HumanAgent, RandomActionAgent

    if name == 'human':
        return HumanAgent()
    elif name == 'dqn':
        if model is None:
            raise ValueError('A model file is required for a DQN agent.')
        return DeepQNetworkAgent(model=model, memory_size=-1, num_last_frames=4, attention=attention)
    elif name == 'random':
        return RandomActionAgent()

    raise KeyError(f'Unknown agent type: "{name}"')


def play_cli(env, agent, num_episodes=10, attention=1):
    """
    Play a set of episodes using the specified Snake agent.
    Use the non-interactive command-line interface and print the summary statistics afterwards.
    
    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
    """

    fruit_stats = []

    print()
    print('Playing:')

    for episode in range(num_episodes):
        timestep, position = env.new_episode()
        agent.begin_episode()
        game_over = False

        while not game_over:
            action = agent.act(timestep.observation, position, timestep.reward, attention)
            env.choose_action(action)
            timestep, position = env.timestep()
            game_over = timestep.is_episode_end

        fruit_stats.append(env.stats.fruits_eaten)

        summary = 'Episode {:3d} / {:3d} | Timesteps {:4d} | Fruits {:2d}'
        print(summary.format(episode + 1, num_episodes, env.stats.timesteps_survived, env.stats.fruits_eaten))

    print()
    print('Fruits eaten {:.1f} +/- stddev {:.1f}'.format(np.mean(fruit_stats), np.std(fruit_stats)))


def play_gui(env, agent, num_episodes, attention=1, visualize=None):
    """
    Play a set of episodes using the specified Snake agent.
    Use the interactive graphical interface.
    
    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
    """

    gui = PyGameGUI()
    gui.load_environment(env)
    gui.load_agent(agent)
    gui.run(num_episodes=num_episodes, attention=attention, visualize=visualize)


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level)
    if parsed_args.attention != -1:
        model = create_vin_model(env, num_last_frames=4, attention=parsed_args.attention)
        model, visualize = visualize_vin_model(env, num_last_frames=4, attention=parsed_args.attention)
        model.load_weights(parsed_args.model)
    else:
        model = load_model(parsed_args.model) if parsed_args.model is not None else None
    
    agent = create_agent(parsed_args.agent, model, parsed_args.attention)

    run_player = play_cli if parsed_args.interface == 'cli' else play_gui
    run_player(env, agent, num_episodes=parsed_args.num_episodes, attention=parsed_args.attention)
    #visualize_cli(env, agent, num_episodes=parsed_args.num_episodes, attention=parsed_args.attention, visualize=visualize_q)
    #play_gui(env, agent, num_episodes=1, attention=parsed_args.attention, visualize=visualize)


if __name__ == '__main__':
    main()
