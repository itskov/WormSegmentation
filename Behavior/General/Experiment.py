import cv2

import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from Behavior.General.ExpDir import ExpDir

from PIL import Image, ImageDraw


class Experiment:
    def __init__(self, expDir):
        self._tracks = None
        self._videoFilename = None
        self._regionsOfInterest = {}
        self._cap = None
        self._numberOfFrames = 0
        self._scale = 1
        self._positions = None
        self._outputDirName = None

        self.initialize(expDir, np.load(expDir.getTracksFile()))

    # Legacy constructor.
    #def __init__(self, videoFilename, tracks):
    #    expDir = ExpDir(path.dirname(videoFilename))

    #    self.initialize(expDir, tracks)

    def initialize(self, expDir, tracks=None):
        print('Initializing Experiment with dir: %s and with %d tracks' % (expDir._expDir, len(tracks)))

        # else leave the tracs as is.
        if (tracks is not None):
            self._tracks = tracks

        self._videoFilename = expDir.getVidFile()
        self._cap = cv2.VideoCapture(self._videoFilename)
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
        self._outputDirName = path.dirname(self._videoFilename)



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



def GenerateHMMData(tracks, id):
    l = np.array([t._trackCords.shape[0] for t in tracks])
    tracks = tracks[l > 750]

    t = tracks[id]
    x = np.diff(t._tracksAngles[1:-2])
    x = np.reshape(x, (-1, 1))

    x2 = t._tracksSpeeds[1:-3]
    x2 = np.reshape(x2, (-1, 1))
    x3 = np.convolve(np.ravel(x), (1,) * 10, "same")
    x3 = np.reshape(x3, (-1, 1))

    X = np.squeeze(np.stack((x, x2, x3), axis=2))
    return(X)


def main():
    from Behavior.Visualizers.RoiAnalysis import RoiAnalysis
    from hmmlearn import hmm

    expDir = ExpDir('/home/itskov/Temp/13-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_16.31.51')

    exp = np.load(expDir.getExpFile())[0]
    tracks = exp._tracks


    model = hmm.GaussianHMM(n_components=2, init_params="tmcs")
    model.transmat_ = np.array([[0.5, 0.5],
                                [0.5, 0.5]])

    X = GenerateHMMData(tracks, 10)
    model.fit(X)
    X = GenerateHMMData(tracks, 11)
    model.fit(X)
    X = GenerateHMMData(tracks, 13)
    model.fit(X)
    X = GenerateHMMData(tracks, 14)
    model.fit(X)

    t = tracks[16]
    X = GenerateHMMData(tracks, 16)
    model.fit(X)
    z = model.predict(X)

    z = np.insert(z, 0,0)
    z = np.insert(z, 0, 0)
    z = np.insert(z, -1,0)
    z = np.insert(z,-1, 0)

    exp.plotTracks([t], [z])


    pass

    #roi = RoiAnalysis(exp)
    #roi.execute()

if __name__ == "__main__":
    main()




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



