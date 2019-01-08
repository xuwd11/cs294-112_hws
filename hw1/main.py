import os
import pickle
import json
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import argparse

from model import Model

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)

DEFAULT_DATA_DIR = os.path.join('.', 'expert_data')
EXPERIMENTS_DIR = os.path.join('.', 'experiments')

# High-level options
parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str)
parser.add_argument('--algorithm', type=str, default='behavioral_cloning', 
                    help='Available algorithms: behavioral_cloning / dagger / all')
parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
parser.add_argument('--mode', type=str, default='train', help='Available modes: all / train / test')
parser.add_argument('--reload', type=str, default='no')
parser.add_argument('--save_name', type=str)
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[100, 100, 100])
parser.add_argument('--loss', type=str, default='l2_loss')


# Hyperparameters for the model
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs to train. 0 means train indefinitely')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--max_gradient_norm', type=float, default=10.0,
                    help='Clip gradients to this norm')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--eval_every', type=int, default=5, 
                    help='How many epochs to do per simulation / dagger')


# Hyperparameters for simulation / evaluation
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=10,
                    help='Number of expert roll outs')


FLAGS = vars(parser.parse_args())
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS['gpu'])


def train_val_split(data, train_size=0.9):
    n = data['observations'].shape[0]
    indices = np.random.permutation(n)
    train_id, val_id = indices[:int(n * train_size)], indices[int(n * train_size):]
    data_train = {'observations': data['observations'][train_id], 'actions': data['actions'][train_id]}
    data_val = {'observations': data['observations'][val_id], 'actions': data['actions'][val_id]}
    return data_train, data_val


def initialize_model(session, model, curr_dir, expect_exists):
    print('Looking for model at {}'.format(curr_dir))
    ckpt = tf.train.get_checkpoint_state(curr_dir)
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print('Reading model parameters from {}'.format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception('There is no saved checkpoint at {}'.format(curr_dir))
        else:
            print('There is no saved checkpoint at {}. Creating model with fresh parameters.'.format(curr_dir))
            session.run(tf.global_variables_initializer())
            print('Num params: {}'.format(sum(v.get_shape().num_elements() for v in tf.trainable_variables())))

            
def train(session, model, curr_dir, bestmodel_dir, data_train, data_val):
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
    
    file_handler = logging.FileHandler(os.path.join(curr_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)
    
    with open(os.path.join(curr_dir, FLAGS['save_name'] + '.json'), 'w') as f:
        json.dump(FLAGS, f)
        
    if not os.path.exists(bestmodel_dir):
        os.makedirs(bestmodel_dir)
    
    initialize_model(session, model, curr_dir, expect_exists=False)
    model.train(session, curr_dir, bestmodel_dir, data_train, data_val)
                
        
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
        FLAGS['save_name'] = FLAGS['env_name'] + '_BC_' + time.strftime('%Y-%m-%d-%H-%M-%S')
    curr_dir = os.path.join(EXPERIMENTS_DIR, FLAGS['save_name'])
    bestmodel_dir = os.path.join(curr_dir, 'best_checkpoint')
    
    with open(os.path.join(DEFAULT_DATA_DIR, FLAGS['env_name'] + '.pkl'), 'rb') as f:
        data = pickle.load(f)
    FLAGS['input_dim'] = data['observations'].shape[-1]
    FLAGS['output_dim'] = data['actions'].shape[-1]
    
    data_train, data_val = train_val_split(data)

    model = Model(FLAGS)
    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    if FLAGS['mode'] == 'train':
        with tf.Session(config=config) as sess:
            train(sess, model, curr_dir, bestmodel_dir, data_train, data_val)

            
if __name__ == '__main__':
    main()