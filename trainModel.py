# @title Training

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import tempfile
import pickle
import shutil
import random

from glob import glob
from os import path
from scipy import io

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import re
import pandas as pd


def cnn_model_fn(origImages_, filteredImages, imageSize):
    vaeBeta = 1
    input_layer = tf.reshape(origImages_, [-1, imageSize[0], imageSize[1], 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        strides = (1,1),
        activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides = (2,2),
        activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides = (2,2),
        activation=tf.nn.relu)


    dconv1 = tf.layers.conv2d_transpose(
        inputs = conv3,
        filters = 32,
        kernel_size = (3,3),
        strides = (2,2),
        padding="same",
        activation=tf.nn.relu)

    dconv2 = tf.layers.conv2d_transpose(
        inputs = dconv1,
        filters = 64,
        kernel_size = (3,3),
        strides = (2,2),
        padding="same",
        activation=tf.nn.relu)

    dconv3 = tf.layers.conv2d(
        inputs = dconv2,
        filters = 128,
        kernel_size = (3,3),
        strides = (1,1),
        padding="same",
        activation=tf.nn.relu)

    output = tf.layers.conv2d(
        inputs = dconv2,
        filters = 1,
        kernel_size = (3,3),
        strides = (1,1),
        padding="same",
        activation=tf.nn.sigmoid)

    loss = tf.reduce_mean(tf.norm(tf.subtract(output, input_layer)))

    return (loss, output)


def getBatch(batchDir, batchNum, imageSize):
    npyFiles = glob(path.join(batchDir, '*.npy'))

    # Chosing the numbers of mats of batch
    chosenFiles = np.random.choice(npyFiles, batchNum)

    origImages = np.zeros((batchNum,) + imageSize)
    filteredImages = np.zeros((batchNum,)+ imageSize)

    for i,fileName in enumerate(chosenFiles):
        currentSample = np.load(fileName)

        origImages[i, :, :] = currentSample[0]
        filteredImages[i, :, :] = currentSample[1]

    return (origImages, filteredImages)


def main():
    # Setting the seed
    # np.random.seed(15574)
    DATA_DIR = '/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/static/TrainData'
    IMAGE_SIZE = (100, 100)
    RESTORE_POINT = "./WormSegmentatioNetworks/WormSegmentation"
    BATCH_SIZE = 50
    N = 2750000
    RESTORE = False

    tf.reset_default_graph()

    origImages_ = tf.placeholder(tf.float64, [None, IMAGE_SIZE[0], IMAGE_SIZE[1]])
    filteredImages_ = tf.placeholder(tf.float64, [None, IMAGE_SIZE[0], IMAGE_SIZE[1]])

    loss, output = cnn_model_fn(origImages_, filteredImages_, IMAGE_SIZE)

    # Solver
    solver = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:

        saver = tf.train.Saver()
        if (RESTORE == False):
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, RESTORE_POINT)

        for i in range(N):
            currentBatch = getBatch(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)

            trainDict = {origImages_: currentBatch[0], filteredImages_: currentBatch[1]}
            sess.run(solver, feed_dict=trainDict)
            lossValue = loss.eval(feed_dict=trainDict);


            if (i % 100 == 0):
                print('\r' + str(i) + ". Loss: " + str(lossValue), end="")
                saver.save(sess, RESTORE_POINT)




main()
