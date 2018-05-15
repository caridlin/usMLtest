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

""" This is adapted from cifar10_input.py. It currently reads txt files but in the
future will be adpated to read fits files for PCWI/KCWI pipeline. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Global constants describing the data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5


def read_fits(filename_queue):
  """Reads txt files both in training and evaluatig data sets.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (1024)
      width: number of columns in the result (1024)
      depth: number of color channels in the result (0)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      fp64image_clean: a [height, width, depth] float64 Tensor with the noise free image data
      fp64image: a [height, width, depth] float64 Tensor with the image data
  """

  class FITSRecord(object):
    pass
  result = FITSRecord()
  
  # Size of the binary file.
  # It's current a 2d array but in the future fits file we have different
  # frequency channels. 
  result.height = 100
  result.width = 100
  result.depth = 1
  image_bytes = 8 * result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = 2 * image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the saved binary file, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of float64 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.float64)
  

  # The clean image part
  depth_minor = tf.reshape(
      tf.strided_slice(record_bytes, [0],
                       [int(image_bytes / 8)]),
      [result.height, result.width, result.depth])
  # Convert from [height, width, depth] to [height, width, depth].
  result.fp64image_clean = tf.transpose(depth_minor, [0, 1, 2])

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [int(image_bytes / 8)],
                       [int(image_bytes / 4)]),
      [result.height, result.width, result.depth])
  # Convert from [height, width, depth] to [height, width, depth].
  result.fp64image = tf.transpose(depth_major, [0, 1, 2])

  return result


def _generate_image_and_label_batch(image, image_clean, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, depth] of type.float64.
    label: 1-D Tensor of type.float64
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, depth] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, images_clean = tf.train.shuffle_batch(
        [image, image_clean],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size,
        min_after_dequeue = min_queue_examples)
  else:
    images, images_clean = tf.train.batch(
        [image, image_clean],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size)


  return images, images_clean


def inputs(eval_data, data_dir, batch_size):
  """Construct input for cnn evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, depths] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
                 for i in xrange(10)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with tf.name_scope('input'):
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_fits(filename_queue)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(read_input.fp64image)

    # Set the shapes of tensors.
    float_image.set_shape([read_input.height, read_input.width, read_input.depth])

    # Similar for clean images
    float_image_clean = tf.image.per_image_standardization(read_input.fp64image_clean)

    float_image_clean.set_shape([read_input.height, read_input.width, read_input.depth])
   
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, float_image_clean,
                                         min_queue_examples, batch_size,
                                         shuffle = True)

