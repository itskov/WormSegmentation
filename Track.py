import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.spatial.distance import  pdist
from scipy.interpolate import interp1d

from time import time

class Track:
    def __init__(self, trackDict):
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
        self._tracksSteps = np.append(self._tracksSteps,[[None,None]],axis=0)



        # Calculate angles
        self._tracksAngles = np.zeros(self._trackCords.shape[0])
        for i in range(self._tracksSteps.shape[0] - 1):
            curStep = self._tracksSteps[i, :]
            self._tracksAngles[i]  = np.arctan2(curStep[0], curStep[1])

        # Calculate reversals
        angDiffs = np.diff(self._tracksAngles[0:-1])
        angDiffs[angDiffs > np.pi] = 2 * np.pi - angDiffs[angDiffs > np.pi]

        self._tracksReversals = abs(angDiffs) > np.pi / 2
        self._tracksReversals = np.insert(self._tracksReversals, 0, 0)
        self._tracksReversals = np.insert(self._tracksReversals, len(self._tracksReversals), 0)


        print('Track created. Time: ' + str(time()  - beforeCreation))



    def getSpeed(self, frame):
        if (frame in self._trackFrames):
            return self._tracksSpeeds[self._trackFrames == frame]
        else:
            return None

    def getReversal(self, frame):
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
            return self._trackCords[self._trackFrames == frame,:]
        else:
            return None


    def getStep(self, frame):
        if (frame in self._trackFrames):
            return self._tracksSteps[self._trackFrames == frame,:]
        else:
            return None

    def plotTrack(self):
        plt.plot(self._trackCords[:,0],self._trackCords[:,1])
        plt.show()


    def isInRange(self, framesRange):
        rangeVals = list(framesRange)

        if (np.max(self._trackFrames) < np.min(rangeVals) or np.min(self._trackFrames) > np.max(rangeVals)):
            return False
        return True




if __name__ == "__main__":
    tracksDicts = np.load('/home/itskov/Temp/tracks2.npy')

    lens = np.asarray([len(list(t.values())) for t in tracksDicts])
    tracksDicts = tracksDicts[lens > 150]

    tracks = [Track(trackDict) for trackDict in tracksDicts]
    tracks = [t for t in tracks if t._maximalDistance > 250]

    frameRange = range(2500,5500)
    speeds = np.zeros((len(list(frameRange)),))
    counts = np.zeros((len(list(frameRange)),))
    reversals = np.zeros((len(list(frameRange)),))
    revCounts = np.zeros((len(list(frameRange)),))


    relevantTracks = [track for track in tracks if track.isInRange(frameRange) == True]
    for ii,i in enumerate(frameRange):
        for j,t in enumerate(relevantTracks):
            curSpeed = t.getSpeed(i)
            curReversal = t.getReversal(i)
            if (curSpeed != None):
                speeds[ii] += curSpeed
                counts[ii] += 1
            if (curReversal != None):
                reversals[ii] += curReversal
                revCounts[ii] += 1

    fig, ax1 = plt.subplots()
    speedsSig = convolve(speeds / counts, np.ones(10,))
    ax1.plot(speedsSig)
    ax2 = ax1.twinx()

    revsSigs = convolve(reversals / revCounts, np.ones(20, ))
    reversalsSig = convolve(revsSigs, np.ones(5, ))
    ax2.plot(reversalsSig, color='r')
    plt.show()

    pass






