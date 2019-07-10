# @title Process Movie
from multiprocessing import Pool

import cv2
import os
import sys

import numpy as np
import tensorflow as tf
import time

import logging


from trainModel import  cnn_model_fn,normalizeFrame

from skvideo.io import FFmpegWriter
from os.path import join


def readFrame(cap, i, height, width):
    #print('Start Reading frame: ' + str(i))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    success, readFrame = cap.read()
    readFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2GRAY)
    readFrame = np.reshape(readFrame, (1, height, width, 1))
    #print('End Reading frame: ' + str(i))
    return (readFrame)


def splitBatch(batchData, bins):
    batchDataSize = batchData.shape


    if (batchDataSize[1] % bins !=0) or (batchDataSize[2] % bins != 0):
        print(batchDataSize)
        print("Error splitting: " + str((batchDataSize[0], batchDataSize[1])) + " do  not divide by " + str(bins))
        return

    print((batchDataSize[1],batchDataSize[2],batchDataSize[0]))
    batchData = np.reshape(batchData, (batchDataSize[1],batchDataSize[2],batchDataSize[0]))
    batchDataSize = batchData.shape

    # Splitting the rows.
    rowSplit = np.split(batchData, bins, axis = 0)
    # Splitting the cols.
    colSplit = np.asarray([np.asarray(np.split(s, bins, axis = 1)) for s in rowSplit])

    # Changing the axis order
    colSplit = np.rollaxis(np.rollaxis(colSplit, axis=3), axis=3)

    splittedBatch = np.reshape(np.asarray(colSplit),
                               (int(batchDataSize[0] / bins), int(batchDataSize[1] / bins), batchDataSize[2] * bins**2))


    splittedBatch = np.reshape(splittedBatch, (splittedBatch.shape[2], splittedBatch.shape[0], splittedBatch.shape[1]))

    return splittedBatch

def mergeBatch(batchData, bins):
    s = batchData.shape
    batchData = batchData.reshape(batchData, (s[1], s[2], s[0]))

    reshaedSplittedData = np.reshape(batchData, (s[0], s[1], int(s[2] / bins ** 2), bins, bins))
    reshaedSplittedData = np.rollaxis(np.rollaxis(reshaedSplittedData, axis=3), axis=4)


    fullData = np.reshape(reshaedSplittedData, (s[0] * bins, s[1] * bins, int(s[2] /  bins ** 2)))

    fullData = np.reshape(fullData, (fullData.shape[2], fullData.shape[0], fullData.shape[1]))

    return(fullData)


def writeLog(logFile, s):
    logFile.writelines([s + os.linesep])
    logFile.flush()

def main():
    if len(sys.argv) != 3:
        print('Usage: processVideo.py <RestorePoint> <FileToProcess> ')
        return

    BINS = 4

    RESTORE_POINT = sys.argv[1]
    INPUT_DIR = os.path.dirname(sys.argv[2])

    logFile = open(join(INPUT_DIR, 'seg.log'),'a')


    writeLog(logFile,'Start segmentation')

    inputFile = os.path.join(INPUT_DIR, 'inputFile.mp4')
    outputFile = os.path.join(INPUT_DIR, 'outputFile.mp4')


    #print('Opening: ' + inputFile)
    writeLog(logFile, 'Opening: ' + inputFile)
    writeLog(logFile, 'Wiring into:' + outputFile)
    cap = cv2.VideoCapture(inputFile)

    if not os.path.exists(inputFile):
        writeLog(logFile, inputFile + ' file do not exist.')

    # Number of frames.
    movieLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print('Video Length:' + str(movieLength))
    writeLog(logFile, 'Video Length:' + str(movieLength))

    # Read first frame to get frame size.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    success, frameRead = cap.read()
    #print('Success opening:' + str(success))
    writeLog(logFile, 'Success opening:' + str(success))

    # Getting the shape of the frame.
    height, width, _ = frameRead.shape

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config = config) as sess:
        currentFrame_ = tf.placeholder(tf.float64, [None, height, width, 1])
        filteredFrame_ = tf.placeholder(tf.float64, [None, height, width, 1])

        loss, output = cnn_model_fn(currentFrame_, filteredFrame_, (int(height / BINS), int(width / BINS)))

        saver = tf.train.Saver()
        saver.restore(sess, RESTORE_POINT)

        videoWriter = FFmpegWriter(outputFile, outputdict={'-crf': '0'})

        batch = 1
        #for i in range(0, movieLength, batch):
        for i in range(350, 380, batch):

            #print('Frame: ' + str(i) + "/" + str(movieLength))
            writeLog(logFile, 'Frame: ' + str(i) + "/" + str(movieLength))

            firstFrame = i
            lastFrame = np.minimum(i + batch, (movieLength - 1))
            framesRead = np.zeros((batch, height, width))

            framesRange = range(firstFrame, lastFrame)
            #print('Reading Frames: ' + str(list(framesRange)))
            writeLog(logFile, 'Reading Frames: ' + str(list(framesRange)))
            beforeRead = time.time()
            for f, j in enumerate(framesRange):
                framesRead[f,:,:] = np.reshape(readFrame(cap, i, height, width), (height, width))


            splittedFramesRead = splitBatch(framesRead, BINS)

            elpsdReading = time.time() - beforeRead
            #print('After Reading. Time: ' + str(elpsdReading))
            writeLog(logFile, 'After Reading. Time: ' + str(elpsdReading))

            framesRead = np.reshape(framesRead, (batch, height, width, 1))
            #print('Start network forward.')
            writeLog(logFile, 'Start network forward.')
            beforeForward = time.time()
            procDict = {currentFrame_: splittedFramesRead, filteredFrame_: splittedFramesRead}
            outputVal = output.eval(procDict)
            outputVal = normalizeFrame(np.reshape(outputVal, (batch, height, width)))

            outputVal = mergeBatch(outputVal, BINS)

            # Binarizing the image.
            outputVal[outputVal < 195] = 0
            outputVal[outputVal >= 195] = 1 * 255

            forwardElpsd = time.time() - beforeForward
            #print('End network forward. Time: ' + str(forwardElpsd))
            writeLog(logFile, 'End network forward. Time: ' + str(forwardElpsd))

            #print('Start writing frame.')
            writeLog(logFile, 'Start writing frame.')
            beforeWriting = time.time()

            for f, j in enumerate(range(firstFrame, lastFrame)):
                videoWriter.writeFrame(outputVal[f, :, :])

            writingElpsd = time.time() - beforeWriting
            #print('After writing. Time: ' + str(writingElpsd))
            writeLog(logFile, 'After writing. Time: ' + str(writingElpsd))



        videoWriter.close()

if __name__ == "__main__":
    main()