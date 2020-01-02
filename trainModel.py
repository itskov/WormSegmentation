# @title Training

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf


from glob import glob
from os import path




def cnn_model_fn(origImages_, filteredImages, imageSize):
    input_layer = tf.reshape(origImages_, [-1, imageSize[0], imageSize[1], 1])
    filtered_images = tf.reshape(filteredImages, [-1, imageSize[0], imageSize[1], 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        strides=(1, 1),
        activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        strides=(2, 2),
        activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=(2, 2),
        activation=tf.nn.relu)

    dconv1 = tf.layers.conv2d_transpose(
        inputs=conv3,
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    dconv1 = tf.concat((dconv1, conv2), axis=3)

    dconv2 = tf.layers.conv2d_transpose(
        inputs=dconv1,
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    dconv2 = tf.concat((dconv2, conv1), axis=3)

    dconv3 = tf.layers.conv2d_transpose(
        inputs=dconv2,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    dconv3 = tf.concat((dconv3, input_layer), axis=3)

    output = tf.layers.conv2d(
        inputs=dconv3,
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same")

    loss = tf.reduce_mean(tf.norm(tf.subtract(output, filtered_images)))

    return loss, output

def getBatch(batchDir, batchNum, imageSize):
    npyFiles = glob(path.join(batchDir, '*.npy'))

    # Chosing the numbers of mats of batch
    chosenFiles = np.random.choice(npyFiles, batchNum)

    origImages = np.zeros((batchNum,) + imageSize)
    filteredImages = np.zeros((batchNum,) + imageSize)

    for i, fileName in enumerate(chosenFiles):
        currentSample = np.load(fileName)

        origImages[i, :, :] = currentSample[0]
        filteredImages[i, :, :] = cv2.blur(currentSample[1], (3, 3))

    return (origImages, filteredImages)


def normalizeFrame(outImage):
    outImage = np.maximum(outImage, 0)
    outImage = np.minimum(outImage, 1)

    #outImage = (outImage - np.min(outImage)) / (np.max(outImage) - np.min(outImage))
    outImage = np.uint8(outImage * 255)
    return (outImage)


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

    origImages_ = tf.placeholder(tf.float32, [None, IMAGE_SIZE[0], IMAGE_SIZE[1]])
    filteredImages_ = tf.placeholder(tf.float32, [None, IMAGE_SIZE[0], IMAGE_SIZE[1]])

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




if __name__ == "__main__":
    main()
