from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(cov_matrices, phase_train=True, reuse=None, weight_decay=0.0, batch_size):
    batch_norm_params={
        'decay':0.995,
        'epsilon':0.001,
        'updates_collections':None,
        'variables_collections':[tf.GraphKeys.TRAINABLE_VARIABLES]
    }
    with slim.arg_scope([slim.fully_connected],
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        return temporal_pool(cov_matrices, is_training=phase_train,batch_size)

def temporal_pool(inputs, is_training=True, reuse=None, scope='TemporalPool',batch_size):
    #dropout_keep_prob=1
    with tf.variable_scope(scope,'TemporalPool',[inputs],reuse=reuse):
      with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
        with tf.variable_scope('spdpooling') as scope:
            shape=inputs.get_shape().as_list()
            # BiRe-1
            weight1, weight2 = _variable_with_orth_weight_decay('orth_weight0', shape,2)
            local6 = tf.matmul(tf.matmul(weight2, inputs), weight1,name='matmulout')
            local7 = _cal_rect_cov(local6)
            # BiRe-2                    
            shape = local7.get_shape().as_list()
            weight3, weight4 = _variable_with_orth_weight_decay('orth_weight1', shape,2)
            local8 = tf.matmul(tf.matmul(weight4, local7), weight3)
            local9 = _cal_rect_cov(local8)
            # BiRe-3                    
            shape = local9.get_shape().as_list()
            print('spdpooling feature2: D1:%d, D2:%d, D3:%d', shape[0], shape[1], shape[2])
            weight5, weight6 = _variable_with_orth_weight_decay('orth_weight2', shape,2)
            local10 = tf.matmul(tf.matmul(weight6, local9), weight5)
            local11 = _cal_rect_cov(local10)
            # BiRe-4
            shape = local11.get_shape().as_list()
            weight7, weight8 = _variable_with_orth_weight_decay('orth_weight3', shape,2)
            local12 = tf.matmul(tf.matmul(weight8, local11), weight7)
            local14=_cal_rect_cov(local12)
            # LogEig Layer
            local13 = _cal_log_cov(local14)
        # The batch size 31 here corresponds to batch size and
        # had to be hard coded while flattening the matrix
        net = tf.reshape(local13,[batch_size,-1])
        net = slim.fully_connected(net,32, activation_fn=None, scope='Bottleneck', reuse=False)
    return net

# Covariance pooling layer
def _cal_cov_pooling(features):
    shape_f = features.get_shape().as_list()
    shape_f = tf.shape(features)
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]), 2)
    centers_batch = tf.reshape(centers_batch, [shape_f[0], 1, shape_f[2]])
    centers_batch = tf.tile(centers_batch, [1, shape_f[1], 1])
    tmp = tf.subtract(features, centers_batch)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    features_t = 1/tf.cast((shape_f[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    trace_t = tf.trace(features_t)
    trace_t = tf.reshape(trace_t, [shape_f[0], 1])
    trace_t = tf.tile(trace_t, [1, shape_f[2]])
    # 0.001 is regularization factor so that the matrix is SPD Matrix
    trace_t = 0.001*tf.matrix_diag(trace_t)
    return tf.add(features_t,trace_t)

'''
# Gaussian pooling layer (alternative to covariance pooling)
def _cal_gaussian_pooling(features):
    shape_f = features.get_shape().as_list()
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]),2)
    print('center batch {}'.format(centers_batch.shape))
    centers_batch = tf.reshape(centers_batch, [shape_f[0], 1, shape_f[2]])
    centers_batch_tile = tf.tile(centers_batch, [1, shape_f[1], 1])
    tmp = tf.subtract(features, centers_batch_tile)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    cov = 1/tf.cast((shape_f[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    cov = tf.add(cov, tf.matmul(tf.transpose(centers_batch,[0,2,1]), centers_batch))
    col_right = tf.reshape(centers_batch, [shape_f[0], shape_f[2], 1])
    new_mat = tf.concat([cov,col_right],2)
    row_bottom = tf.concat([centers_batch,tf.ones([shape_f[0],1,1])],2)
    features_t = tf.concat([new_mat,row_bottom],1)
    shape_f = features_t.get_shape().as_list()
    trace_t = tf.trace(features_t)
    trace_t = tf.reshape(trace_t, [shape_f[0], 1])
    trace_t = tf.tile(trace_t, [1, shape_f[2]])
    trace_t = 0.001*tf.matrix_diag(trace_t)
    return tf.add(features_t,trace_t)
'''

# LogEig Layer
def _cal_log_cov(features):
    [s_f, v_f] = tf.self_adjoint_eig(features)
    s_f = tf.log(s_f)
    s_f = tf.matrix_diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t


# computes weights for BiMap Layer
def _variable_with_orth_weight_decay(name1, shape,red):
    s1 = tf.cast(shape[2], tf.int32)
    s2 = tf.cast(shape[2]/red, tf.int32)
    w0_init, _ = tf.qr(tf.random_normal([s1, s2], mean=0.0, stddev=1.0))
    w0 = tf.get_variable(name1, initializer=w0_init)
    tmp1 = tf.reshape(w0, (1, s1, s2))
    tmp2 = tf.reshape(tf.transpose(w0), (1, s2, s1))
    tmp1 = tf.tile(tmp1, [shape[0], 1, 1])
    tmp2 = tf.tile(tmp2, [shape[0], 1, 1])
    return tmp1, tmp2

#
def _cal_rect_cov(features):
    [s_f, v_f] = tf.self_adjoint_eig(features)
    s_f = tf.clip_by_value(s_f, 1e-4, 10000)
    s_f = tf.matrix_diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t
