import os
import pickle
import json
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import argparse

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_DATA_DIR = os.path.join('.', 'expert_data')
EXPERIMENTS_DIR = os.path.join('.', 'experiments')

# High-level options
parser = argparse.ArgumentParser()
parser.add_argument('data_name', type=str)
parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
parser.add_argument('--mode', type=str, default='all', help='Available modes: all / train / test')
parser.add_argument('--reload', type=str, default='no')
parser.add_argument('--save_name', type=str)

# Hyperparameters
parser.add_argument('--num_epochs', type=int, default=0,
                    help='Number of epochs to train. 0 means train indefinitely')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--max_gradient_norm', type=float, default=10.0,
                    help='Clip gradients to this norm')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[100, 100, 100])

FLAGS = vars(parser.parse_args())

def main():
    if (not FLAGS['mode'] in ['all', 'train']) or FLAGS['reload'] == 'yes':
        if FLAGS['save_name'] is None:
            raise ValueError('Please enter save_name for reloading')
        with open(os.path.join(EXPERIMENTS_DIR, FLAGS['save_name'], 
                               FLAGS['save_name'] + '.json'), 'r') as f:
            saved_flags = json.load(f)
            for name, value in saved_flags.items():
                if not name in ['mode', 'save_name']:
                    FLAGS[name] = value
        print('Old flags have been reloaded successfully.')
    elif FLAGS['save_name'] is None:
        FLAGS['save_name'] = FLAGS['data_name'] + '_BC_' + time.strftime('%Y-%m-%d-%H-%M-%S')
    curr_dir = os.path.join(EXPERIMENTS_DIR, FLAGS['save_name'])
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
    with open(os.path.join(DEFAULT_DATA_DIR, FLAGS['data_name'] + '.pkl'), 'rb') as f:
        data = pickle.load(f)
    FLAGS['input_dim'] = data['observations'].shape[-1]
    FLAGS['output_dim'] = data['actions'].shape[-1]
    with open(os.path.join(curr_dir, FLAGS['save_name'] + '.json'), 'w') as f:
        json.dump(FLAGS, f)

if __name__ == '__main__':
    main()