#!/usr/bin/env python3.6

""" Front-end script for training a Snake agent. """

import json
import sys

from keras.models import Sequential
from keras.layers import *
from keras.layers.merge import *
from keras.optimizers import *

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser

from vin_keras.vin import vin_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--attention',
        required=True,
        type=int,
        help='whether to use attentino.',
    )
    
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=30000,
        help='The number of episodes to run consecutively.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=0)


def create_vin_model(env, num_last_frames, attention=1):
    """
    Build a new DQN model to be used for training.
    
    Args:
        env: an instance of Snake environment. 
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """


    image_shape = (num_last_frames, ) + env.observation_shape
    print(image_shape)

    vin = vin_model(l_s=image_shape[2], 
                    k = 20, 
                    l_q=3,
                    l_a=8,
                    attention=attention)
    vin.summary()

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

    return merged_model


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.
    
    Args:
        env: an instance of Snake environment. 
        num_last_frames: the number of last frames the agent considers as state.
    Returns:
        A compiled DQN model.
    """

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
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env.num_actions))

    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level)
    if parsed_args.attention == -1:
        model = create_dqn_model(env, num_last_frames=4)
    else:
        model = create_vin_model(env, num_last_frames=4, attention=parsed_args.attention)
    #model.load_weights('dqn-final.model')
    agent = DeepQNetworkAgent(
        model=model,
        memory_size=-1,
        num_last_frames=4,
        attention=parsed_args.attention
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.95
    )
    '''agent.train(
        env,
        batch_size=64,
        num_episodes=3000,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.95,
        exploration_range=(0.1,0.1)
    )'''


if __name__ == '__main__':
    main()
