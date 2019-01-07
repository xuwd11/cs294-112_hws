import os
import logging

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

class Model:
    '''Top-level model'''
    
    def __init__(self, FLAGS):
        print('Initializing the model...')
        self.FLAGS = FLAGS
        
        with tf.variable_scope(
            'model', 
            initializer=tf.keras.initializers.he_normal(), 
            regularizer=tf.contrib.layers.l2_regularizer(scale=3e-7), 
            reuse=tf.AUTO_REUSE
        ):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()
    
    def add_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.FLAGS['input_dim']])
        self.y = tf.placeholder(tf.float32, shape=[None, 1, self.FLAGS['output_dim']])
        
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        
    def build_graph(self):
        out = tf.contrib.layers.fully_connected(self.x, self.FLAGS['hidden_dims'][0], tf.nn.relu, scope='h0')
        for i, n in enumerate(self.FLAGS['hidden_dims'][1:]):
            out = tf.contrib.layers.fully_connected(out, self.FLAGS['hidden_dims'][n], tf.nn.relu, scope='h{}'.format(i + 1))
            out = tf.nn.dropout(out, self.keep_prob)
        out = tf.contrib.layers.fully_connected(out, self.FLAGS['output_dim'], activation_fn=None, scope='final')
        self.out = tf.expand_dims(out, axis=1)
    
    def add_loss(self):
        with tf.variable_scope('loss'):
            if self.FLAGS['loss'] == 'l2_loss':
                self.loss = tf.nn.l2_loss(self.y - self.out)
        