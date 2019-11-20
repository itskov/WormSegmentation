import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.spatial.distance import  pdist
from scipy.interpolate import interp1d

from time import time

class Track:
    def __init__(self, trackDict):
        self._trackDict = trackDict
        beforeCreation = time()
        trackFrames = np.array(list(trackDict.keys()))
        trackCords = np.array(list(trackDict.values()))

        # Convolving the track to smooth it.
        convKerel = np.ones((3,)) * (1/3)
        self._trackCordsSmoothed = np.zeros(trackCords.shape)
        xs = convolve(trackCords[:, 0], convKerel, mode='valid')
        ys = convolve(trackCords[:, 1], convKerel, mode='valid')
        xs = np.insert(xs, 0, xs[0])
        xs = np.append(xs, xs[-1])
        ys = np.insert(ys, 0, ys[0])
        ys = np.append(ys, ys[-1])


        if not np.all(np.diff(trackFrames) == 1):
            interpXs = interp1d(trackFrames, xs)
            interpYs = interp1d(trackFrames, ys)

            self._trackFrames = np.array(range(np.min(trackFrames), np.max(trackFrames) + 1))

            finalXs = [interpXs(i) for i in self._trackFrames]
            finalYs = [interpYs(i) for i in self._trackFrames]

            self._trackCords = np.zeros((len(finalXs),) + (2,))
            self._trackCords[:, 0] = finalXs
            self._trackCords[:, 1] = finalYs
        else:
            self._trackFrames = trackFrames
            self._trackCords = np.zeros((len(xs),) + (2,))
            self._trackCords[:, 0] = xs
            self._trackCords[:, 1] = ys


        # Calculate speeds
        self._tracksSteps = np.diff(self._trackCords, axis = 0)
        self._tracksSpeeds = np.sqrt(self._tracksSteps[:,0] ** 2 + self._tracksSteps[:,1] ** 2)
        self._tracksSpeeds = np.append(self._tracksSpeeds, 0)
        # Adding one last fictive step.
        self._tracksSteps = np.append(self._tracksSteps,[[None, None]],axis=0)
        self._tracksSteps = self._tracksSteps.astype(np.float32)


        # Calculate angles
        self._tracksAngles = np.zeros(self._trackCords.shape[0])
        for i in range(self._tracksSteps.shape[0] - 1):
            curStep = self._tracksSteps[i, :]
            self._tracksAngles[i] = np.arctan2(curStep[0], curStep[1])

        # Calculate reversals
        angDiffs = np.diff(self._tracksAngles[0:-1])
        angDiffs[angDiffs > np.pi] = 2 * np.pi - angDiffs[angDiffs > np.pi]

        self._tracksReversals = abs(angDiffs) > (np.pi / 2.5)
        self._tracksReversals = np.insert(self._tracksReversals, 0, 0)
        self._tracksReversals = np.insert(self._tracksReversals, len(self._tracksReversals), 0)


        print('Track created. Time: ' + str(time()  - beforeCreation))


    # NOT READY
    def getTrackSegment(self, pos, distanceThr, isBigger):
        distances = np.linalg.norm(self._trackCords - pos, axis=1)

        if isBigger:
            intPoses = distances > distanceThr
        else:
            intPoses = distances < distanceThr

        interestingPosStart = np.argmin(np.where(intPoses))
        interestingPosEnd = np.argmax(np.where(np.logical_not(intPoses[interestingPosStart:])))


        return self


    def getTrackPirouettesMark(self):
        FILTER_SIZE = 35
        filter = list((0,) + tuple(np.ones((FILTER_SIZE - 1,))))
        firstConv = np.convolve(self._tracksReversals, filter, "same")
        secondConv = np.convolve(np.flip(self._tracksReversals), filter, "same")

        pirMark = (firstConv > 1) & (secondConv > 1)

        return pirMark


    def getMeanProjection(self, pos):
        beforeDistance = np.linalg.norm(self._trackCords[0, :] - pos)
        afterDistance = np.linalg.norm(self._trackCords[-1, :] - pos)

        deltaDistance = beforeDistance - afterDistance
        return (deltaDistance / self._trackCords.shape[0])

    def getMeanSpeed(self):
        return np.mean(self._tracksSpeeds)


    def getDistances(self,  pos):
        distances = np.linalg.norm(np.array(pos) - self._trackCords, axis=1)
        return distances

    def getAngles(self, pos):
        firstVecs = pos - self._trackCords
        secondVecs = self._tracksSteps.astype(np.float32)

        angles = np.zeros(self._tracksSteps.shape[0] - 1)
        for i in range(firstVecs.shape[0] - 1):
            dotProd = np.dot(firstVecs[i,:], secondVecs[i,:])
            dotProd /= np.linalg.norm(firstVecs[i, :]) * np.linalg.norm(secondVecs[i, :])
            angles[i] = np.arccos(dotProd)


        angles = np.append(angles, [None], axis=0)

        return angles

    def getSpeed(self, frame):
        if (frame in self._trackFrames):
            return self._tracksSpeeds[self._trackFrames == frame]
        else:
            return None
    def getAbsAngles(self, frame):
        if (frame in self._trackFrames):
            rev =  self._tracksAngles[self._trackFrames == frame]
            if rev == np.NaN:
                return None
            else:
                return rev
        else:
            return None

    def getPos(self, frame):
        if (frame in self._trackFrames):
            return self._trackCords[self._trackFrames == frame, :]
        else:
            return None


    def getStep(self, frame):
        if (frame in self._trackFrames):
            return self._tracksSteps[self._trackFrames == frame, :]
        else:
            return None

    def plotTrack(self):
        plt.plot(self._trackCords[:,0],self._trackCords[:, 1])
        plt.show()


    def isInRange(self, framesRange):
        rangeVals = list(framesRange)

        if np.max(self._trackFrames) < np.min(rangeVals) or np.min(self._trackFrames) > np.max(rangeVals):
            return False
        return True

    def getMaxDistTravelled(self):
        return np.max(pdist(self._trackCords))

if __name__  == "__main__":
    exp = np.load('/mnt/storageNASRe/tph1/Results/12-Sep-2019/TPH_1_ATR_TRAIN_IAA3.avi_12.14.20/exp.npy')[0]
    tracks = exp._tracks

    from Behavior.Visualizers.AngPosDensity import AngPosDensity


    #ap = AngPosDensity(exp)
    #ap.execute()

    l = [track._trackCords.shape[0] for track in tracks]
    tracks = tracks[np.array(l) > 500]


    t = tracks[186]
    t2 = t.getTrackSegment(exp._regionsOfInterest['endReg']['pos'], 250, True)

    plt.figure();
    exp.plotTracks([t]);
    plt.figure();
    plt.plot(t.getAngles(exp._regionsOfInterest['endReg']['pos']))
    #plt.show()


    pass






