# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os
import scipy.misc
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf


# functions
def _conv_layer(input, weights, bias, name):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias, name=name)


def _pool_layer(input, name):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=(1, 2, 2, 1), padding='SAME', name=name)


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2',
        'pool1',
        'conv2_1', 'relu2_1',
        'conv2_2', 'relu2_2',
        'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
        'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4',
        'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
        'pool5',
        'fc1',
        'relu_fc1'
    )
    data = loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))  # 实际上就是 dtype('<f8') [ 123.68 ,  116.779,  103.939]，固定的
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, [1, 0, 2, 3])
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias, name=name)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = _pool_layer(current, name=name)
        else:
            kernels, bias = weights[37][0][0][0][0]
            kernels = np.transpose(kernels, [1, 0, 2, 3])
            bias = bias.reshape(-1)
            conv = tf.nn.conv2d(current, kernels, strides=(1, 1, 1, 1), padding='VALID', name=name)
            current = tf.nn.bias_add(conv, bias, name=name)
        net[name] = current
    assert len(net) == len(layers)
    del data
    return net, mean_pixel, layers


if __name__ == '__main__':
    # path
    cwd = os.getcwd()
    VGG_PATH = cwd + '\\imagenet-vgg-verydeep-19.mat'
    IMG_PATH = cwd + '\\pic\\dogetest.jpg'
    input_image = imread(IMG_PATH)
    shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    print('perparations are ready.')

    # sess
    with tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        nets, mean_pixel, all_layers = net(VGG_PATH, image)
        input_image_pre = np.array([preprocess(input_image, mean_pixel)])
        layers = all_layers
        for i, layer in enumerate(layers):
            print('[%d/%d] %s' % (i + 1, len(layers), layer))
            features = nets[layer].eval(session=sess, feed_dict={image: input_image_pre})
            print("dtype:{} shape: {}, ".format(features.dtype, features.shape))
            if i == 38:
                saveShape = np.reshape(features, [4096])
                np.savetxt('resultV.txt', saveShape)
