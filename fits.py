# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" This is adapted from cifar10.py. It builds the cnn network. 

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
from numpy import sqrt
import tensorflow as tf

import fits_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 2,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp32', True,
                            """Train the model using fp32.""")

# Global constants describing the data set.
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
IMAGE_DEPTH = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = fits_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = fits_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999      # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0        # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.002  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.02     # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32 if FLAGS.use_fp32 else tf.float64
    var = tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
  return var


def _variable_with_weight_decay(name, shape, mean, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32 if FLAGS.use_fp32 else tf.float64
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(mean = mean, stddev = stddev, dtype = dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inputs(eval_data):
  """Construct input for evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, depth] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  images, images_clean = fits_input.inputs(eval_data = eval_data,
                                        data_dir = data_dir,
                                        batch_size = FLAGS.batch_size)
  if FLAGS.use_fp32:
    images = tf.cast(images, tf.float32)
    images_clean = tf.cast(images_clean, tf.float32)
  return images, images_clean


def inference(images):
  """Build the cnn model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    norm3: which is the deconvolved image.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 1, 8],
                                         mean = -0.6,
                                         stddev=1/sqrt(5*5*1),
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [8], tf.constant_initializer([-4.15, 1.8, 4.3]))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.sigmoid(pre_activation, name=scope.name)
    _activation_summary(conv1)


  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 8, 8],
                                         mean = 0.5,
                                         stddev=1/sqrt(5*5*8),
                                         wd=None)
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [8], tf.constant_initializer([-9, -8, 6.6]))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.sigmoid(pre_activation, name=scope.name)
    _activation_summary(conv2)

##
##  # conv3
##  with tf.variable_scope('conv3') as scope:
##    kernel = _variable_with_weight_decay('weights',
##                                         shape=[5, 5, 8, 8],
##                                         mean = 0.15,
##                                         stddev=1/sqrt(5*5*8),
##                                         wd=None)
##    conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
##    biases = _variable_on_cpu('biases', [8], tf.constant_initializer([-2.8, 1]))
##    pre_activation = tf.nn.bias_add(conv, biases)
##    conv3 = tf.nn.sigmoid(pre_activation, name=scope.name)
##    _activation_summary(conv3)


##  # conv4
##  with tf.variable_scope('conv4') as scope:
##    kernel = _variable_with_weight_decay('weights',
##                                         shape=[5, 5, 8, 8],
##                                         mean = 0.15,
##                                         stddev=1/sqrt(5*5*8),
##                                         wd=None)
##    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
##    biases = _variable_on_cpu('biases', [8], tf.constant_initializer([-3, -2, 1]))
##    pre_activation = tf.nn.bias_add(conv, biases)
##    conv4 = tf.nn.sigmoid(pre_activation, name=scope.name)
##    _activation_summary(conv4)


  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 8, 1],
                                         mean = 0,
                                         stddev=1/sqrt(5*5*8),
                                         wd=0.004)
    conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [1],  tf.constant_initializer(6.2))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.sigmoid(pre_activation, name=scope.name)
    _activation_summary(conv5)

  #image_predict = conv5
  image_predict = tf.stack([tf.image.per_image_standardization(conv5[index]) for index in range(FLAGS.batch_size)])
  return image_predict

### conv1
##  with tf.variable_scope('conv1') as scope:
##    kernel = _variable_with_weight_decay('weights',
##                                         shape=[5, 5, 1, 8],
##                                         stddev=1/sqrt(5*5*1),
##                                         wd=None)
##    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
##    biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.1))
##    pre_activation = tf.nn.bias_add(conv, biases)
##    conv1 = tf.nn.relu(pre_activation, name=scope.name)
##    _activation_summary(conv1)
##
##  # norm1
##  norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=1, beta=0.5,
##                    name='norm1')
##
##
##  # conv2
##  with tf.variable_scope('conv2') as scope:
##    kernel = _variable_with_weight_decay('weights',
##                                         shape=[5, 5, 8, 8],
##                                         stddev=1/sqrt(5*5*8),
##                                         wd=None)
##    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
##    biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.1))
##    pre_activation = tf.nn.bias_add(conv, biases)
##    conv2 = tf.nn.relu(pre_activation, name=scope.name)
##    _activation_summary(conv2)
##
##  # norm2
##  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=1, beta=0.5,
##                    name='norm2')
##
##  # conv3
##  with tf.variable_scope('conv3') as scope:
##    kernel = _variable_with_weight_decay('weights',
##                                         shape=[5, 5, 8, 8],
##                                         stddev=1/sqrt(5*5*8),
##                                         wd=None)
##    conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
##    biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.1))
##    pre_activation = tf.nn.bias_add(conv, biases)
##    conv3 = tf.nn.relu(pre_activation, name=scope.name)
##    _activation_summary(conv3)
##
##  # norm3
##  norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=1, beta=0.5,
##                    name='norm3')
##
##  # conv4
##  with tf.variable_scope('conv4') as scope:
##    kernel = _variable_with_weight_decay('weights',
##                                         shape=[5, 5, 8, 8],
##                                         stddev=1/sqrt(5*5*8),
##                                         wd=None)
##    conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='SAME')
##    biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.1))
##    pre_activation = tf.nn.bias_add(conv, biases)
##    conv4 = tf.nn.relu(pre_activation, name=scope.name)
##    _activation_summary(conv4)
##
##  # norm4
##  norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=1, beta=0.5,
##                    name='norm4')
##
##  # conv5
##  with tf.variable_scope('conv5') as scope:
##    kernel = _variable_with_weight_decay('weights',
##                                         shape=[5, 5, 8, 1],
##                                         stddev=1/sqrt(5*5*8),
##                                         wd=0.004)
##    conv = tf.nn.conv2d(norm4, kernel, [1, 1, 1, 1], padding='SAME')
##    biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
##    pre_activation = tf.nn.bias_add(conv, biases)
##    conv5 = tf.nn.relu(pre_activation, name=scope.name)
##    _activation_summary(conv5)
##
##  # norm5
##  norm5 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=1, beta=0.5,
##                    name='norm5')
##
##
##  return norm5


def loss(images_predict, images_clean):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the images to 1D
  batch_size = FLAGS.batch_size
  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH
  depth = IMAGE_DEPTH
  images_predict = tf.reshape(images_predict, [batch_size, height * width * depth])
  images_clean =tf.reshape(images_clean, [batch_size, height * width * depth])
  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.l2_loss(images_predict - images_clean, name='cross_entropy_per_example')
  cross_entropy_mean = cross_entropy / batch_size
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss_eval(images_predict, images_clean):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the images to 1D
  batch_size = FLAGS.batch_size
  height = IMAGE_HEIGHT
  width = IMAGE_WIDTH
  depth = IMAGE_DEPTH
  images_predict = tf.reshape(images_predict, [batch_size, height * width * depth])
  images_clean =tf.reshape(images_clean, [batch_size, height * width * depth])
  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.l2_loss(images_predict - images_clean, name='cross_entropy_per_example_eval')
  cross_entropy_mean = cross_entropy / batch_size

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return cross_entropy_mean


def _add_loss_summaries(total_loss):
  """Add summaries for losses in the cnn model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train the cnn model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    #opt = tf.train.AdadeltaOptimizer(lr)
    #opt = tf.train.AdagradOptimizer(lr)
    opt = tf.train.AdamOptimizer(lr)
    #opt = tf.train.ProximalGradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op

