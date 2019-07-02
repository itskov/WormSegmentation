# @title Process Movie
from multiprocessing import Pool

import cv2
import os
import sys

import numpy as np
import tensorflow as tf
import time



from trainModel import  cnn_model_fn,normalizeFrame

from skvideo.io import FFmpegWriter


def readFrame(cap, i, height, width):
    #print('Start Reading frame: ' + str(i))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    success, readFrame = cap.read()
    readFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2GRAY)
    readFrame = np.reshape(readFrame, (1, height, width, 1))
    #print('End Reading frame: ' + str(i))
    return (readFrame)



def main():
    if len(sys.argv) != 3:
        print('Usage: processVideo.py <RestorePoint> <FileToProcess> ')
        return

    RESTORE_POINT = sys.argv[1]
    INPUT_DIR = os.path.abspath(sys.argv[2])



    inputFile = os.path.join(INPUT_DIR, 'inputFile.mp4')
    outputFile = os.path.join(INPUT_DIR, 'outputFile.mp4')


    print('Opening: ' + inputFile)
    print('Wiring into:' + outputFile)
    cap = cv2.VideoCapture(inputFile)

    if not os.path.exists(inputFile):
        print(inputFile + ' file do not exist.')

    # Number of frames.
    movieLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video Length:' + str(movieLength))

    # Read first frame to get frame size.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    success, frameRead = cap.read()
    print('Success opening:' + str(success))

    # Getting the shape of the frame.
    height, width, _ = frameRead.shape

    tf.reset_default_graph()

    with tf.Session() as sess:
        currentFrame_ = tf.placeholder(tf.float64, [None, height, width, 1])
        filteredFrame_ = tf.placeholder(tf.float64, [None, height, width, 1])

        loss, output = cnn_model_fn(currentFrame_, filteredFrame_, (height, width))

        saver = tf.train.Saver()
        saver.restore(sess, RESTORE_POINT)

        videoWriter = FFmpegWriter(outputFile, outputdict={'-crf': '0'})

        batch = 3
        for i in range(0, movieLength, batch):
            print('Frame: ' + str(i) + "/" + str(movieLength))

            firstFrame = i
            lastFrame = i + batch
            framesRead = np.zeros((batch, height, width))

            framesRange = range(firstFrame, lastFrame)
            print('Reading Frames: ' + str(list(framesRange)))
            for f, j in enumerate(framesRange):
                framesRead[f,:,:] = np.reshape(readFrame(cap, i, height, width), (height, width))


            framesRead = np.reshape(framesRead, (batch, height, width, 1))
            print('Start network forward.')
            beforeForward = time.time()
            procDict = {currentFrame_: framesRead, filteredFrame_: framesRead}
            outputVal = output.eval(procDict)
            outputVal = normalizeFrame(np.reshape(outputVal, (height, width)))
            forwardElpsd = time.time() - beforeForward
            print('End network forward. Time: ' + str(forwardElpsd))

            print('Start writing frame.')
            beforeWriting = time.time()

            for f, j in enumerate(range(firstFrame, lastFrame)):
                videoWriter.writeFrame(framesRead[f, :, :])

            writingElpsd = time.time() - beforeWriting
            print('After writing. Time: ' + str(writingElpsd))


        videoWriter.close()

if __name__ == "__main__":
    main()