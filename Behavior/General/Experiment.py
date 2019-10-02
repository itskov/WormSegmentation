import cv2

import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from Behavior.General.ExpDir import ExpDir

from PIL import Image, ImageDraw


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

        # Here we save directory
        self._outputDirName = path.dirname(videoFilename)

    def __setstate__(self, d):
        self.__dict__ = d
        # Since we cannot serialize the cv2 object.
        self._cap = cv2.VideoCapture(self._videoFilename)
        print('Reinitializing.')

    def takeScale(self):
        success, sampleFrame = self._cap.read()
        plt.imshow(sampleFrame);
        newPoints = plt.ginput(2, timeout=-1)
        plt.close()

        self._scale = np.linalg.norm(np.array(newPoints[0]) - np.array(newPoints[1]))


    def plotTracks(self, tracks, marks=None):
        success, sampleFrame = self._cap.read()

        plt.imshow(sampleFrame)
        for i,track in enumerate(tracks):
            if marks is not None and marks[i] is not None \
                    and marks[i].shape[0] == track._trackCords.shape[0]:
                mark = marks[i]
                for j in range(track._trackCords.shape[0] - 1):
                    plt.plot(track._trackCords[j:(j+2),1], track._trackCords[j:(j + 2),0],
                             color='blue' if mark[j] == 0 else 'yellow')
            else:
                plt.plot(track._trackCords[:, 1], track._trackCords[:, 0], color='blue')

            plt.scatter(track._trackCords[0, 1], track._trackCords[0, 0], color='red', s=50)
            plt.scatter(track._trackCords[-1, 1], track._trackCords[-1, 0], color='green', s=50)

        plt.show()


    def getFrameSize(self):
        success, sampleFrame = self._cap.read()
        if (success == False):
            print('Error reading from: ' + self._videoFilename)

        return (sampleFrame.shape[0:2])

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

    def save(self):
        self._cap = None
        np.save(path.join(self._outputDirName, 'exp'), [self], allow_pickle=True)


if __name__ == "__main__":
    from glob2 import glob
    from Behavior.General.Track import Track

    expDirs = glob('/home/itskov/Temp/Chris/17-Mar-2019_Chris/*')

    for curDir in expDirs:
        print(curDir)
        try:
            expDir = ExpDir(curDir)
            exp = np.load(expDir.getExpFile())[0]
            #exp.takeScale()

            coolTracks = exp._tracks[np.array([track._trackCords.shape[0] for track in exp._tracks]) > 400]
            t = coolTracks[0].getTrackSegment(exp._regionsOfInterest['startReg']['pos'], 450, True)

            t = coolTracks[190]
            exp.plotTracks([t], [t.getTrackPirouettesMark()])

            #exp.addCirclePotisionRad('startReg', exp._scale / 2)
            #exp.addCirclePotisionRad('endReg', exp._scale / 4)
            #exp.save()
        except Exception as exp:
            print('Error: ' + str(exp))
            continue


'''''    from RoiAnalysis import RoiAnalysis

    expDir = ExpDir('/home/itskov/Temp/05-Sep-2019/TPH_1_ATR_TRAIN_NO_IAA3.avi_20.48.41')

    rawFile = expDir.getVidFile()
    tracks = np.load(expDir.getTracksFile())


    exp = Experiment(rawFile, tracks)
    exp.takeScale()
    exp.addCirclePotisionRad('startReg', exp._scale / 2)
    exp.addCirclePotisionRad('endReg', exp._scale / 4)
    exp.save()

    roiAtr = RoiAnalysis(exp)
    roiAtr.execute()

    expDir = ExpDir('/home/itskov/Temp/05-Sep-2019/TPH_1_NO_ATR_TRAIN_NO_IAA3.avi_20.44.37')
    rawFile = expDir.getVidFile()
    tracks = np.load(expDir.getTracksFile())



    exp = Experiment(rawFile, tracks)
    exp.takeScale()

    exp.addCirclePotisionRad('startReg', exp._scale / 2)
    exp.addCirclePotisionRad('endReg', exp._scale / 4)
    exp.save()

    roiNoAtr = RoiAnalysis(exp)
    roiNoAtr.execute()

    import matplotlib.pyplot as plt;
    seaborn.set()
    plt.close('all')
    plt.plot(roiNoAtr._results['arrivedFrac']);
    plt.plot(roiAtr._results['arrivedFrac']);
    plt.show()
    pass; '''



