import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import convolve

class Track:
    def __init__(self, trackDict):
        self._trackFrames = np.array(list(trackDict.keys()))
        self._trackCords = np.array(list(trackDict.values()))

        # Calculate speeds
        tracksSteps = np.diff(self._trackCords, axis = 0)
        self._tracksSpeeds = np.sqrt(tracksSteps[:,0] ** 2 + tracksSteps[:,1] ** 2)
        self._tracksSpeeds = np.insert(self._tracksSpeeds, 0, 0)


        # Calculate angles
        self._tracksAngles = np.zeros(self._trackFrames.shape)
        for i in range(tracksSteps.shape[0] - 1):
            curStep = tracksSteps[i, :]
            cosAng = np.dot(curStep, [1,0]) / (np.linalg.norm(curStep))

            # between 0 and pi radians
            self._tracksAngles[i] = np.arccos(cosAng)


        # Calculate reversals
        angDiffs = np.diff(self._tracksAngles[0:-1])
        self._tracksReversals = abs(angDiffs) > np.pi/2.2
        self._tracksReversals = np.insert(self._tracksReversals, 0, 0)
        self._tracksReversals = np.insert(self._tracksReversals, len(self._tracksReversals), 0)

    def getSpeed(self, frame):
        if (frame in self._trackFrames):
            return self._tracksSpeeds[self._trackFrames == frame]
        else:
            return None

    def getReversal(self, frame):
        if (frame in self._trackFrames):
            rev =  self._tracksAngles[self._trackFrames == frame]
            if (rev == np.NaN):
                return None
            else:
                return rev
        else:
            return None


    def isInRange(self, framesRange):
        rangeVals = list(framesRange)

        if (np.max(self._trackFrames) < np.min(rangeVals) or np.min(self._trackFrames) > np.max(rangeVals)):
            return False
        return True




if __name__ == "__main__":
    tracksDicts = np.load('/home/itskov/Temp/tracks2.npy')

    lens = np.asarray([len(list(t.values())) for t in tracksDicts])
    tracksDicts = tracksDicts[lens > 75]

    tracks = [Track(trackDict) for trackDict in tracksDicts]

    frameRange = range(1500,3500)
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
    reversalsSig = convolve(reversals / revCounts, np.ones(5, ))
    ax2.plot(reversalsSig, color='r')
    plt.show()

    pass






