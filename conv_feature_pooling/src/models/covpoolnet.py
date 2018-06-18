from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=7, weight_decay=0.0, reuse=None, batch_size=128):
    batch_norm_params={
        'decay':0.995,
        'epsilon':0.001,
        'updates_collections':None,
        'variables_collections':[tf.GraphKeys.TRAINABLE_VARIABLES]
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        return network(images, is_training=phase_train,dropout_keep_prob=keep_probability,bottleneck_layer_size=bottleneck_layer_size, batch_size)

def network(inputs,is_training=True,dropout_keep_prob=1,bottleneck_layer_size=7, reuse=None, scope='CovPoolNetwork', batch_size=128):
    #dropout_keep_prob=1
    with tf.variable_scope(scope,'CovPoolNetwork',[inputs],reuse=reuse):
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                #1
                net = slim.conv2d(inputs, 64, 3, stride=1, padding='SAME', scope='Conv2d_1')
                net = tf.nn.relu(net)
                net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='MaxPool_1')
                #net=tf.Print(net,[net],message="Local5 Tensor")
                #4
                net = slim.conv2d(net, 96, 3, stride=1, padding='SAME', scope='Conv2d_2')
                net = tf.nn.relu(net)
                net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='MaxPool_2')
                
                #7
                net = slim.conv2d(net, 128, 3, stride=1, padding='SAME', scope='Conv2d_3')
                net = tf.nn.relu(net)
                
                net = slim.conv2d(net, 128, 3, stride=1, padding='SAME', scope='Conv2d_4')
                net = tf.nn.relu(net)
                net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='MaxPool_3')
                
                #12
                net = slim.conv2d(net, 256, 3, stride=1, padding='SAME', scope='Conv2d_5')
                print('Conv2d_4: {}'.format(net.shape))
                net = tf.nn.relu(net)
                
                #14
                net = slim.conv2d(net, 256, 3, stride=1, padding='SAME', scope='Conv2dfo_6')
                net = tf.nn.relu(net)
                             
                with tf.variable_scope('spdpooling1') as scope:
                    shape = net.get_shape().as_list()
                    reshaped = tf.reshape(net, [shape[0], shape[1]*shape[2], shape[3]])
                    # Cov Pooling Layer
                    local5 = _cal_cov_pooling(reshaped)
                    print('Name {}'.format(local5.shape))
                    shape = local5.get_shape().as_list()
                    # BiRe Layer - 1
                    weight1, weight2 = _variable_with_orth_weight_decay('orth_weight0', shape)
                    local6 = tf.matmul(tf.matmul(weight2, local5), weight1,name='matmulout')
                    local7 = _cal_rect_cov(local6)
		    '''
                    # Additional BiRe Layer
                    shape = local7.get_shape().as_list()
                    print('spdpooling feature2: D1:%d, D2:%d, D3:%d', shape[0], shape[1], shape[2])
                    weight1, weight2 = _variable_with_orth_weight_decay('orth_weight1', shape)
                    local8 = tf.matmul(tf.matmul(weight2, local7), weight1)                        
                    '''
		    local9 = _cal_log_cov(local7)                    
                    
                net = tf.reshape(local9,[batch_size,-1])
                net = slim.fully_connected(net, 2000, activation_fn=None, scope='fc_1', reuse=False)
                net = tf.nn.relu(net)
                net = slim.fully_connected(net, 128, activation_fn=None, scope='fc_1', reuse=False)
                net = tf.nn.relu(net)
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
    return net

'''
implementation of covariance pooling layers
'''
def _cal_cov_pooling(features):
    shape_f = features.get_shape().as_list()
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]),2)
    centers_batch = tf.reshape(centers_batch, [shape_f[0], 1, shape_f[2]])
    centers_batch = tf.tile(centers_batch, [1, shape_f[1], 1])
    tmp = tf.subtract(features, centers_batch)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    features_t = 1/tf.cast((shape_f[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    trace_t = tf.trace(features_t)
    trace_t = tf.reshape(trace_t, [shape_f[0], 1])
    trace_t = tf.tile(trace_t, [1, shape_f[2]])
    trace_t = 0.001*tf.matrix_diag(trace_t)
    return tf.add(features_t,trace_t)

# Implementation is of basically LogEig Layer
def _cal_log_cov(features):
    [s_f, v_f] = tf.self_adjoint_eig(features)
    s_f = tf.log(s_f)
    s_f = tf.matrix_diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t

# computes weights for BiMap Layer
def _variable_with_orth_weight_decay(name1, shape):
    s1 = tf.cast(shape[2], tf.int32)
    s2 = tf.cast(shape[2]/2, tf.int32)
    w0_init, _ = tf.qr(tf.random_normal([s1, s2], mean=0.0, stddev=1.0))
    w0 = tf.get_variable(name1, initializer=w0_init)
    tmp1 = tf.reshape(w0, (1, s1, s2))
    tmp2 = tf.reshape(tf.transpose(w0), (1, s2, s1))
    tmp1 = tf.tile(tmp1, [shape[0], 1, 1])
    tmp2 = tf.tile(tmp2, [shape[0], 1, 1])
    return tmp1, tmp2

# ReEig Layer
def _cal_rect_cov(features):
    [s_f, v_f] = tf.self_adjoint_eig(features)
    s_f = tf.clip_by_value(s_f, 0.0001, 10000)
    s_f = tf.matrix_diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t
