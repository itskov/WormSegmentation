import cv2

import numpy as np
import matplotlib.pyplot as plt
from ExpDir import ExpDir

from PIL import Image, ImageDraw
from Track import Track

from AngleVisualizer import AngleVisualizer
from SegmentedTracker import SegmentedTracker


class Experiment:
    def __init__(self, videoFilename, tracks=None):
        self._tracks = tracks
        self._videoFilename = videoFilename
        self._cap = cv2.VideoCapture(videoFilename)
        self._regionsOfInterest = {}

        # Getting the length of the movie.
        self._numberOfFrames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setting the movie to be in the first frame ( a length process ).
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Take a scale so we can compare between different experimental settings.
        self._scale = 1

        # Here we store important positions.
        self._positions = {}


    def takeScale(self):
        success, sampleFrame = self._cap.read()
        plt.imshow(sampleFrame);
        newPoints = plt.ginput(2, timeout=-1)
        plt.close()

        self._scale = np.linalg.norm(np.array(newPoints[0]) - np.array(newPoints[1]))



    def addCirclePotisionRad(self, pointName, rad):
        success, sampleFrame = self._cap.read()
        plt.imshow(sampleFrame);
        center = (plt.ginput(1, timeout=-1)[0])


        plt.close()

        # Drawing a circle around the chem point.
        sampleImage = Image.fromarray(sampleFrame).convert('RGB')
        imageDraw = ImageDraw.Draw(sampleImage)
        imageDraw.arc((center[0] - rad, center[1] - rad, center[0] + rad, center[1] + rad),
                      0,
                      360,
                      fill='red')

        plt.imshow(np.array(sampleImage))

        # Saving the point.
        newRegion = {'pos' : (center[1], center[0]), 'rad' : rad}
        self._regionsOfInterest[pointName] = newRegion



    def addCirclePosition(self, pointName):
        success, sampleFrame = self._cap.read()
        plt.imshow(sampleFrame);
        newPoints = plt.ginput(2, timeout=-1)
        plt.close()

        rad = np.linalg.norm(np.diff(newPoints, axis=0))
        center = newPoints[0]


        # Drawing a circle around the chem point.
        sampleImage = Image.fromarray(sampleFrame).convert('RGB')
        imageDraw = ImageDraw.Draw(sampleImage)
        imageDraw.arc((center[0] - rad, center[1] - rad, center[0] + rad, center[1] + rad),
                      0,
                      360,
                      fill='red')

        plt.imshow(np.array(sampleImage))

        # Saving the point.
        newRegion = {'pos' : np.fliplr(newPoints[0]), 'rad' : newPoints[1]}
        self._regionsOfInterest[pointName] = newRegion






if __name__ == "__main__":
    from RoiAnalysis import RoiAnalysis

    expDir = ExpDir('/mnt/storageNASRe/tph1/25-Aug-2019/TPH_1_ATR_TRAIN_IAA3.avi_14.38.38/')

    rawFile = expDir.getVidFile()
    tracks = np.load(expDir.getTracksFile())


    exp = Experiment(rawFile, tracks)
    exp.takeScale()

    exp.addCirclePotisionRad('startReg', exp._scale / 5)
    exp.addCirclePotisionRad('endReg', exp._scale / 4)

    roiAtr = RoiAnalysis(exp)
    roiAtr.execute()

    expDir = ExpDir('/mnt/storageNASRe/tph1/25-Aug-2019/TPH_1_NO_ATR_TRAIN_IAA2.avi_14.39.28/')
    rawFile = expDir.getVidFile()
    tracks = np.load(expDir.getTracksFile())



    exp = Experiment(rawFile, tracks)
    exp.takeScale()

    exp.addCirclePotisionRad('startReg', exp._scale / 5)
    exp.addCirclePotisionRad('endReg', exp._scale / 4)

    roiNoAtr = RoiAnalysis(exp)
    roiNoAtr.execute()

    import matplotlib.pyplot as plt;
    plt.close('all')
    plt.plot(roiNoAtr._results['arrivedFrac']);
    plt.plot(roiAtr._results['arrivedFrac']);
    plt.show()
    pass;



