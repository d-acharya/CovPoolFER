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
# modified by d-acharya

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import tempPoolFramework
import os
import sys
import math
import glob
import itertools

def compute_cov_matrix(video_path):
    video_path_split=video_path.split('/')
    video_path_split=video_path_split[-1]
    array=np.load(os.path.join(video_path, video_path_split+'.npy'))
    feature_list=array
    return tempPoolFramework.cov_computation(feature_list)

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            dataset = tempPoolFramework.get_video_dataset(args.data_dir)
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]
            paths, labels = tempPoolFramework.get_video_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of videos: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            tempPoolFramework.load_model(args.model)
            
            # Get input and output tensors
            cov_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            logits = tf.get_default_graph().get_tensor_by_name("Logits/BiasAdd:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            logit_size = logits.get_shape()[1]
            prediction = tf.nn.softmax(logits)

            # Run forward pass to calculate embeddings
            print('Calculating features for videos')
            nrof_videos = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_videos / args.batch_size))
            emb_array = np.zeros((nrof_videos , logit_size))
            count = 0
            best_class_indices=np.zeros((nrof_videos))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_videos)
                count = count+(end_index-start_index)
                paths_batch = paths[start_index:end_index]
                if (end_index-start_index)!=args.batch_size:
                    paths_batch.extend(paths[0:(args.batch_size-(end_index-start_index))])
                    #print(paths_batch)
                cov_matrices=np.zeros((args.batch_size,128,128))
                fc = 0
                for filename in paths_batch:
                    cov_matrices[fc,:,:]=compute_cov_matrix(filename)
                    fc = fc + 1
                feed_dict = { cov_placeholder:cov_matrices, phase_train_placeholder:False }
                arr = sess.run(prediction, feed_dict=feed_dict)
                if (end_index-start_index)!=args.batch_size:
                    arr=arr[0:end_index-start_index]
                best_class_indices[start_index:end_index]= np.argmax(arr, axis=1)

            print("Samples considered: {}. Total Videos: {}".format(count, nrof_videos))

            # Classify extracted features
            print('Testing classifier')
            accuracy = np.mean(np.equal(best_class_indices, labels))
            print('Total Accuracy: %.3f%%' % (accuracy*100.0))
            
            # normalized accuracy (on face detection by MTCNN)
            # 383 is total videos in afew validation set
            norm_acc = ((100.0*accuracy*float(nrof_videos))+(1.0/7.0)*(383.-float(nrof_videos)))/383.0
            print('Total Accuracy accounting for face detection failure: %.3f%%' % (norm_acc))

            np.set_printoptions(precision=2)
            # compute accuracy for each class:
            classes=np.unique(labels)
            classes=np.zeros(len(classes))
            classCount = np.zeros(len(classes))
            tCount = 0
            for i in labels:
                classes[i]=classes[i]+np.equal(best_class_indices[tCount], i)
                tCount = tCount + 1
                classCount[i] = classCount[i]+1
            
            for i in range(len(classCount)):
                classes[i]=classes[i]/classCount[i]
            print('Per Class Accuracy: ')
            accuracy=0
            for i in range (len(classCount)):
                print('Accuracy of {} {}%'.format(class_names[i],100*classes[i]))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned AFEW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int,
        help='Number of videos to process in a batch.', default=90)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_videos_per_class', type=int,
        help='Only include classes with at least this number of videos in the dataset', default=20)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
