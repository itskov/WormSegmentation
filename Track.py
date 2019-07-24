import numpy as np

import matplotlib.pyplot as plt

class Track:
    def __init__(self, trackDict):


        self._trackFrames = np.array(list(trackDict.keys()))
        self._trackCords = np.array(list(trackDict.values()))

        # Calculate speeds
        tracksSteps = np.diff(self._trackCords, axis = 0)
        self._tracksSpeeds = np.sqrt(tracksSteps[:,0] ** 2 + tracksSteps[:,1] ** 2)
        self._tracksSpeeds = np.insert(self._tracksSpeeds, 0, 0)

    def getSpeed(self, frame):
        if (frame in self._trackFrames):
            return self._tracksSpeeds[self._trackFrames == frame]
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

    frameRange = range(2000,5000)
    speeds = np.zeros((len(list(frameRange)),))
    counts = np.zeros((len(list(frameRange)),))

    relevantTracks = [track for track in tracks if track.isInRange(frameRange) == True]
    for ii,i in enumerate(frameRange):
        for j,t in enumerate(relevantTracks):
            curSpeed = t.getSpeed(i)
            if (curSpeed != None):
                speeds[ii] += curSpeed
                counts[ii] += 1

    plt.plot(speeds / counts); plt.show()
    pass






