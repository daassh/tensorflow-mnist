# -*- coding: utf-8 -*-
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def model_number(x, keep_prob, is_training=True):
    return convolutional(x, keep_prob, 51, is_training)
     
    
def convolutional(x, keep_prob, label_length, is_training=True):

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([3, 3, 1, 256])
    b_conv1 = bias_variable([256])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_bn1   = tf.contrib.layers.batch_norm(h_conv1, is_training=is_training)

    W_conv2 = weight_variable([3, 3, 256, 128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_bn1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)  # 14 * 14
    
    W_conv3 = weight_variable([3, 3, 128, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_bn3   = tf.contrib.layers.batch_norm(h_conv3, is_training=is_training)

    W_conv4 = weight_variable([3, 3, 128, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_bn3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)  # 7 * 7
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool4_flat = tf.reshape(h_pool4, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, label_length])
    b_fc2 = bias_variable([label_length])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, W_fc2, b_fc2]
    
def convolutional_right_wrong(x, keep_prob):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])


    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]