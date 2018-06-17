"""Functions for building the face recognition network.
"""
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

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile

def compute_cov_matrix_from_csv_np(video_path):
    video_path_split=video_path.decode().split('/')
    video_path_split=video_path_split[-1]
    array=np.load(os.path.join(video_path.decode(), video_path_split+'.npy'))
    size = np.shape(array)[0]
    # either select all frames, 
    # or select 60% frames randomly or centered at center
    min_len=min(size, 10)
    ratio = .6
    r=random.randint(0,2)
    if r==0:
        subset_index=range(0,size)
    elif r==1:
        subset_size = max(np.int32(ratio * size), min_len)
        subset_index=random.sample(range(0, size), subset_size-1)
    else:
        subset_size=max(np.int32(ratio * size), min_len)
        origin_of_subclip = np.int32(np.ceil(size/2-subset_size/2))
        noise = random.randint(-origin_of_subclip,origin_of_subclip)
        perturbed_origin = origin_of_subclip+noise
        subset_index = range(perturbed_origin, perturbed_origin+subset_size-1)
    feature_list=[array[i] for i in subset_index]
    return cov_computation(feature_list)

def cov_computation(feature_list):
    features = feature_list
    shape_=np.shape(features)
    centers = np.mean(features,axis=0)
    tmp = np.subtract(features, centers)
    tmp_t = np.transpose(tmp)
    features_t = 1./(shape_[0]-1.)*np.matmul(tmp_t, tmp)
    trace_t = np.trace(features_t)
    trace_t = np.tile(trace_t, [shape_[1]])
    trace_t = 1e-3*np.diag(trace_t)
    ret = np.add(features_t,trace_t)
    ret = ret.astype(np.float32)
    return ret

'''
# This can be used for gaussian pooling
def gaussian_computation(feature_list):
    features = feature_list
    shape_=np.shape(features)
    centers = np.mean(features,axis=0)
    tmp = np.subtract(features, centers)
    tmp_t = np.transpose(tmp)
    features_t = 1./(shape_[0]-1.)*np.matmul(tmp_t, tmp)
    centers_t=np.transpose(centers)
    cov = np.add(features_t,np.outer(centers,centers_t))
    col_right = centers_t
    new_mat = np.concatenate([cov,np.asarray([col_right]).T],axis=1)
    row_bottom = centers
    row_bottom = np.concatenate([row_bottom,[1]],axis=0)
    features_t = np.concatenate([new_mat,[row_bottom]],axis=0)
    shape_=np.shape(features_t)
    trace_t=np.trace(features_t)
    trace_t = np.tile(trace_t, [shape_[1]])
    trace_t = 1e-3*np.diag(trace_t)
    features_t = np.add(features_t,trace_t)
    features_t = features_t.astype(np.float32)
    return features_t
'''
  
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
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
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def trainspd(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    for idx, (egrad, var) in enumerate(grads):
        if 'orth' in var.name:
            # print('var1.name:%s', var.name)
            #egrad=tf.Print(egrad,[egrad],'RGradient: Before'+var.name)
            tmp1 = tf.matmul(tf.transpose(var), egrad)
            tmp2 = 0.5 * (tmp1 + tf.transpose(tmp1))
            rgrad = egrad - tf.matmul(var, tmp2)
            #rgrad=tf.Print(rgrad,[rgrad],'RGradient: After'+var.name)
            grads[idx] = (rgrad, var)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # stiefel update
    redun = 0.
    for grad, var in grads:
        if 'orth' in var.name:
            o_n, _ = tf.qr(var)
            redun = redun + tf.reduce_sum(var.assign(o_n), [0, 1])

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
   
    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op, redun

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate

class VideoClass():
    "Stores the paths to videos for a given class"
    def __init__(self, name, video_paths):
        self.name = name
        self.video_paths = video_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.video_paths)) + ' videos'
  
    def __len__(self):
        return len(self.video_paths)

def get_video_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                videos = os.listdir(facedir)
                videos = [os.path.join(facedir, vid_name) for vid_name in videos]
                videos = [vid for vid in videos if os.path.isdir(vid)]
                dataset.append(VideoClass(class_name, videos))
    return dataset

def get_video_paths_and_labels(dataset):
    video_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        video_paths_flat += dataset[i].video_paths
        labels_flat += [i] * len(dataset[i].video_paths)
    return video_paths_flat, labels_flat

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
