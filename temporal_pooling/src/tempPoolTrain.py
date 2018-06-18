# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# this code is largely based on framework of facenet.
# modified by d-acharya for covariance pooling

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import os
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import io
import tempPoolFramework
import tempPoolNetwork

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

def main(args):
	subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
	log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
	if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
		os.makedirs(log_dir)
	model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
	if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
		os.makedirs(model_dir)
	np.random.seed(seed=args.seed)
	random.seed(args.seed)
	train_set = tempPoolFramework.get_video_dataset(args.data_dir)
	nrof_classes = len(train_set)
	print('Model directory: %s' % model_dir)
	print('Log directory: %s' % log_dir)

	pretrained_model = None
	if args.pretrained_model:
		pretrained_model = os.path.expanduser(args.pretrained_model)
		print('Pre-trained model: %s' % pretrained_model)

	with tf.Graph().as_default():
		tf.set_random_seed(args.seed)
		global_step = tf.Variable(0, trainable=False)
        
    # Get a list of video paths and their labels
		vid_list, label_list = tempPoolFramework.get_video_paths_and_labels(train_set)
		assert len(vid_list)>0, 'The dataset should not be empty'
        	# Create a queue that produces indices into the video_list and label_list 
		labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
		range_size = array_ops.shape(labels)[0]
		index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)

		index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
		learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

		batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
		phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
		video_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='video_paths')

		labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
        
		input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)
		enqueue_op = input_queue.enqueue_many([video_paths_placeholder, labels_placeholder], name='enqueue_op')
		nrof_preprocess_threads = 4
		videos_and_labels = []
		for _ in range(nrof_preprocess_threads):
			foldernames, label = input_queue.dequeue()
			cov_matrices = []
			for foldername in tf.unstack(foldernames):
				cov_mat = tf.py_func(tempPoolFramework.compute_cov_matrix_from_csv_np,[foldername],tf.float32)
				cov_mat.set_shape((128,128))
				cov_matrices.append(cov_mat)
			videos_and_labels.append([cov_matrices, label])

		video_batch, label_batch = tf.train.batch_join(
			# get batch
			videos_and_labels, batch_size=batch_size_placeholder, 
			shapes=[(128, 128), ()], enqueue_many=True,
			capacity=4 * nrof_preprocess_threads * args.batch_size,
			allow_smaller_final_batch=False)
		print(tf.shape(video_batch))
		video_batch = tf.reshape(video_batch,[args.batch_size,128,128])
		video_batch = tf.identity(video_batch, 'video_batch')
		video_batch = tf.identity(video_batch, 'input')
		label_batch = tf.identity(label_batch, 'label_batch')
        
		print('Total number of classes: %d' % nrof_classes)
		print('Total number of examples: %d' % len(vid_list))
        
		print('Building training graph')
		prelogits = tempPoolNetwork.inference(video_batch, phase_train=phase_train_placeholder, weight_decay=args.weight_decay, batch_size=args.batch_size)
		logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='Logits', reuse=False)

   	#Add center loss
		if args.center_loss_factor>0.0:
			prelogits_center_loss, _ = tempPoolFramework.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
			tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

		learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
			args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
		tf.summary.scalar('learning_rate', learning_rate)

   	# Calculate the average cross entropy loss across the batch
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=label_batch, logits=logits, name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
        

    # Calculate the total losses
		regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
    # Build a Graph that trains the model with one batch of examples and updates the model parameters
		train_op, redun = tempPoolFramework.trainspd(total_loss, global_step, args.optimizer, 
			learning_rate, args.moving_average_decay, tf.global_variables())
        
   	# Create a saver
		saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

   	# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.summary.merge_all()

   	# Start running operations on the Graph.
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
		sess=tf.Session(config=config)
		#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		coord = tf.train.Coordinator()
		tf.train.start_queue_runners(coord=coord, sess=sess)

		with sess.as_default():
			if pretrained_model:
				print('Restoring pretrained model: %s' % pretrained_model)
				saver.restore(sess, pretrained_model)
			print('Running training')
			epoch = 0
			while epoch < args.max_nrof_epochs:
				step = sess.run(global_step, feed_dict=None)
				epoch = step // args.epoch_size
     		# Train for one epoch
				train(args, sess, epoch, vid_list, label_list, index_dequeue_op, enqueue_op, video_paths_placeholder, labels_placeholder,
					learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
					total_loss, train_op, redun, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)
				save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
	sess.close()
	return model_dir

def train(args, sess, epoch, video_list, label_list, index_dequeue_op, enqueue_op, video_paths_placeholder, labels_placeholder, 
	learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
	loss, train_op, redun, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
	batch_number = 0

	if args.learning_rate>0.0:
		lr = args.learning_rate
	else:
		lr = tempPoolFramework.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

	index_epoch = sess.run(index_dequeue_op)
	label_epoch = np.array(label_list)[index_epoch]
	video_epoch = np.array(video_list)[index_epoch]
    
    	# Enqueue one epoch of video paths and labels
	labels_array = np.expand_dims(np.array(label_epoch),1)
	video_paths_array = np.expand_dims(np.array(video_epoch),1)
	sess.run(enqueue_op, {video_paths_placeholder: video_paths_array, labels_placeholder: labels_array})

 	# Training loop
	train_time = 0
	while batch_number < args.epoch_size:
		start_time = time.time()
		feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
		if (batch_number % 100 == 0):
			err, _, _, step, reg_loss, summary_str = sess.run([loss, train_op, redun, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
			summary_writer.add_summary(summary_str, global_step=step)
		else:
			err, _, _, step, reg_loss = sess.run([loss, train_op, redun,  global_step, regularization_losses], feed_dict=feed_dict)
		duration = time.time() - start_time
		print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
			(epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
		batch_number += 1
		train_time += duration
    	# Add validation loss and accuracy to summary
	summary = tf.Summary()
    	#pylint: disable=maybe-no-member
	summary.value.add(tag='time/total', simple_value=train_time)
	summary_writer.add_summary(summary, step)
	return step

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    	# Save the model checkpoint
	print('Saving variables')
	start_time = time.time()
	checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
	saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
	save_time_variables = time.time() - start_time
	print('Variables saved in %.2f seconds' % save_time_variables)
	metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
	save_time_metagraph = 0  
	if not os.path.exists(metagraph_filename):
		print('Saving metagraph')
		start_time = time.time()
		saver.export_meta_graph(metagraph_filename)
		save_time_metagraph = time.time() - start_time
		print('Metagraph saved in %.2f seconds' % save_time_metagraph)
	summary = tf.Summary()
    	#pylint: disable=maybe-no-member
	summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
	summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
	summary_writer.add_summary(summary, step)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--logs_base_dir', type=str, 
	 help='Directory where to write event logs.', default='~/logs/temporalPoolFramework')
	parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/tempPoolFramework')
	parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
	parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
	parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/afew_128')
	parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=400)
	parser.add_argument('--batch_size', type=int,
        help='Number of videos to process in a batch.', default=90)
	parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
	parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
	parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
	parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
	parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
	parser.add_argument('--weight_decay', type=float, 
        help='L2 weight regularization.', default=0.0)
	parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
	parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
	parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
	parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
	parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
	return parser.parse_args(argv)
  

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
