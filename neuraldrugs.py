#!/usr/bin/python

import tensorflow as tf
import numpy as np
import math
import sys
import random
import glob
import os
import argparse

def print_names(model_file):
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_file + ".meta")
    new_saver.restore(sess, model_file)

    for v in tf.trainable_variables():
      print(v.name)
      print(v.get_shape())

def drug(model_file, name_var_weights, vol = 0.5, val = False):

  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_file + ".meta")
    new_saver.restore(sess, model_file)

    if name_var_weights:
      w = [v for v in tf.trainable_variables() if v.name == name_var_weights][0]
    else:
      print "WARNING! It's very dangereous to use this tool without defining weights variable name"
      w = [v for v in tf.trainable_variables()][0]

    v = sess.run(w)
    d = 1 # element dimension
    if isinstance(v[0,0], (list, tuple, np.ndarray)):
      d = len(v[0,0])

    ttl_n = len(v)*len(v[0])*d
    vol = int(ttl_n*float(vol)/100)
    max_weight = np.max(np.max(v, axis=1), axis=0)
    min_weight = np.min(np.min(v, axis=1), axis=0)

    if not isinstance(min_weight, (int, float, long)):
      min_weight = min(min_weight)
      max_weight = max(max_weight)

    for i in range(0, vol):
      k = random.randint(0,len(v)-1)
      j = random.randint(0,len(v[0])-1)
      if isinstance(v[k,j], (int, float, long)):
        if val:
          v[ k,j ] = val#min_weight
        else:
          v[ k,j ] = random.uniform(min_weight, max_weight)
      else:
        l = random.randint(0, len(v[k,j])-1)
        if val:
          v[ k,j ][ l ] = val#min_weight
        else:
          v[ k,j ][ l ] = random.uniform(min_weight, max_weight)

    sess.run(w.assign(v))
    new_saver.save(sess, model_file)

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Neuraldrugs project')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to corrupt')
    parser.add_argument(
        '--name_var_weights',
        type=str,
        default=False,
        help='Name of the variable with weights. I.e. --name-var-weights logits/weights:0 for the im2txt network')
    parser.add_argument(
        '--dosage',
        type=float,
        default=0.01,
        help='How many neurons would be corrupted'
        'in percents 0-100%. Default: 0.01%')
#    parser.add_argument(
#        '--set_weights_fix_value',
#        type=float,
#        default=0.00,
#        help='To set this fixed value'
#        'of weigth for each corrupted neuron')
    parser.add_argument(
        '--set_weights_random_value',
        action='store_true',
        default=True,
        help='Set the random value for neurons'
        'between minimum and maximum values. LSD model')
#    parser.add_argument(
#        '--set_weights_min_value',
#        action='store_true',
#        default=False,
#        help='Set the minimum value for all neurons'
#        'Alcohol model')
    parser.add_argument(
        '--set_weights_max_value',
        action='store_true',
        default=False,
        help='Set the maximum value for all neurons'
        'MDA model')
    parser.add_argument(
        '--print_variables_names',
        action='store_true',
        default=False,
        help='Print variables names from the model'
        'and exit'
    )

    return parser.parse_args()

args = get_arguments()
path = args.checkpoint
vol = args.dosage
val = args.set_weights_fix_value
if args.set_weights_random_value:
    val = False

if args.print_variables_names:
    print_variables_names(path)
else:
    drug(path, args.name_var_weights, vol, val) #40% of neurons will be removed
