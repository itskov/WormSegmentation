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
        self._tracksSteps = np.diff(self._trackCords, axis=0)
        self._tracksSpeeds = np.sqrt(self._tracksSteps[:, 0] ** 2 + self._tracksSteps[:, 1] ** 2)
        self._tracksSpeeds = np.append(self._tracksSpeeds, 0)
        # Adding one last fictive step.
        self._tracksSteps = np.append(self._tracksSteps, [[None, None]], axis=0)
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


        #print('Track created. Time: ' + str(time()  - beforeCreation))

    '''
    Creating a new track dictionary. Some procedures expect
    the track to be a dict.
    '''
    def getTrackDict(self):
        track_dict = dict(zip(self._trackFrames, self._trackCords))
        return track_dict

    def trimTrack(self, endFrame, startFrame=0):
        keepCords = np.bitwise_and(self._trackFrames >= startFrame, self._trackFrames <= endFrame)

        newFrame = self._trackFrames[keepCords]
        newCords = self._trackCords[keepCords, :]

        newCordsDict = dict(zip(newFrame, newCords))

        if np.sum(keepCords) > 5:
            return Track(newCordsDict)

        return None


    def getTrackPirouettesMark(self):
        FILTER_SIZE = 35
        filter = list((0,) + tuple(np.ones((FILTER_SIZE - 1,))))
        firstConv = np.convolve(self._tracksReversals, filter, "same")
        secondConv = np.convolve(np.flip(self._tracksReversals), filter, "same")

        pirMark = (firstConv > 1) & (secondConv > 1)

        return pirMark

    def getRunsLength(self):
        runs_length = np.diff(np.where(self._tracksReversals))
        return runs_length

    def getMeanProjection(self, pos):
        beforeDistance = np.linalg.norm(self._trackCords[0, :] - pos)
        afterDistance = np.linalg.norm(self._trackCords[-1, :] - pos)

        deltaDistance = beforeDistance - afterDistance
        return (deltaDistance / np.sum(self._tracksSpeeds))

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
            rev = self._tracksAngles[self._trackFrames == frame][0]
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
    exp_atr = np.load('/home/itskov/Temp/behav/TPH_1_ATR_TRAIN_75M_0D.avi_14.20.27/exp.npy')[0]
    exp_noAtr = np.load('/home/itskov/Temp/behav/TPH_1_NO_ATR_TRAIN_75M_D0.avi_14.19.35/exp.npy')[0]


    from Behavior.General.TracksFilter import filterTracksForAnalyses
    #tracks_atr = filterTracksForAnalyses(exp._atr._tracks, minSteps=50, minDistance=50)
    #tracks_noAtr = filterTracksForAnalyses(exp_noAtr._atr._tracks, minSteps=50, minDistance=50)
    pass

    from Behavior.Visualizers.RunsLengthAnalyses import RunsLengthAnalyses
    runs_vis_atr = RunsLengthAnalyses(exp_atr)
    runs_vis_atr.execute()

    runs_vis_no_atr = RunsLengthAnalyses(exp_noAtr)
    runs_vis_no_atr.execute()

    import pandas as pd
    df_atr = pd.DataFrame({'len': runs_vis_atr._results['run_lens'], 'cond': 'ATR+'})
    df_no_atr = pd.DataFrame({'len': runs_vis_no_atr._results['run_lens'], 'cond': 'ATR-'})
    df = pd.concat((df_atr, df_no_atr))





    plt.hist(runs_vis._results['run_lens'], bins=1000)


    #ap = AngPosDensity(exp)
    #ap.execute()

    #l = [track._trackCords.shape[0] for track in tracks]
    #tracks = tracks[np.array(l) > 500]


    t = tracks[186]
    t2 = t.getTrackSegment(exp._regionsOfInterest['endReg']['pos'], 250, True)

    plt.figure();
    exp.plotTracks([t]);
    plt.figure();
    plt.plot(t.getAngles(exp._regionsOfInterest['endReg']['pos']))
    #plt.show()


    pass






