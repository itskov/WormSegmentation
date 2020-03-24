from Behavior.General.TracksFilter import filterTracksForAnalyses
from time import time

import numpy as np

import pandas as pd

def checkCorrelation(filename):
    exp = np.load(filename)[0]
    tracks = exp._tracks

    # Filtering tracks
    tracks = filterTracksForAnalyses(tracks, minSteps=25, minDistance=50)

    angles = {}
    print("Calculating angles..")
    bearings = [tracks[i].getAngles(exp._regionsOfInterest['endReg']['pos']) for i in range(len(tracks))]
    bearingsDict = dict(zip(range(len(tracks)), bearings))


    #df = pd.DataFrame({'trackId1': [], 'trackId2': [], 'angle1': [],
    #                   'bearing1': [], 'angle2': [],'bearing2': [],
    #                   'distance': [], 'angDistance': [], 'frame': []})

    trackId1 = []
    trackId2 = []
    angle1 = []
    bearing1 = []
    angle2 = []
    bearing2 = []
    distances = []
    angDistances = []
    frame = []
    distsFromStart1 = []
    distsFromEnd1 = []
    distsFromStart2 = []
    distsFromEnd2 = []

    for i, t in enumerate(tracks):
        print('Going over track: %d / %d. Len: %d' % (i, len(tracks), t._trackFrames.shape[0]))
        before = time()
        tBearings = bearingsDict[i]


        for f in t._trackFrames:
            t1CurrentBearing = tBearings[t._trackFrames == f][0]
            tCurrentAngle = t.getAbsAngles(f)

            distFromStart1 = np.linalg.norm(
                t._trackCords[t._trackFrames == f] - exp._regionsOfInterest['startReg']['pos'])
            distFromEnd1 = np.linalg.norm(
                t._trackCords[t._trackFrames == f] - exp._regionsOfInterest['endReg']['pos'])


            if t1CurrentBearing is None or np.isnan(t1CurrentBearing):
                continue

            for j in range((i+1), len(tracks)):
                t2 = tracks[j]
                if f in t2._trackFrames:
                    t2Bearings = bearingsDict[j]
                    t2CurrentBearing = t2Bearings[t2._trackFrames == f][0]
                    t2CurrentAngle = t2.getAbsAngles(f)

                    if t2CurrentBearing is None or np.isnan(t2CurrentBearing):
                        continue

                    distFromStart2 = np.linalg.norm(
                        t2._trackCords[t2._trackFrames == f] - exp._regionsOfInterest['startReg']['pos'])
                    distFromEnd2 = np.linalg.norm(
                        t2._trackCords[t2._trackFrames == f] - exp._regionsOfInterest['endReg']['pos'])

                    vec = t._trackCords[t._trackFrames == f] - t2._trackCords[t2._trackFrames == f]
                    distance = np.sqrt(np.sum(vec ** 2))
                    '''dicts.append({'trackId1': i,
                                    'trackId2': j,
                                    'angle1': tCurrentAngle,
                                    'bearing1': t1CurrentBearing,
                                    'angle2': t2CurrentAngle,
                                    'bearing2': t2CurrentBearing,
                                    'distance': distance,
                                    'angDistance': np.min((tCurrentAngle - t2CurrentAngle, 2*np.pi - (tCurrentAngle - t2CurrentAngle))),
                                    'frame': f})'''
                    trackId1.append(i)
                    trackId2.append(j)
                    angle1.append(tCurrentAngle)
                    bearing1.append(t1CurrentBearing)
                    angle2.append(t2CurrentAngle)
                    bearing2.append(t2CurrentBearing)
                    distances.append(distance)
                    angDistances.append(np.min((tCurrentAngle - t2CurrentAngle, 2*np.pi - (tCurrentAngle - t2CurrentAngle))))
                    distsFromStart1.append(distFromStart1)
                    distsFromEnd1.append(distFromEnd1)
                    distsFromStart2.append(distFromStart2)
                    distsFromEnd2.append(distFromEnd2)
                    frame.append(f)

            after = time() - before
        print('Time: %f seconds. Entries: %d' % (after, len(trackId1)))

    df = pd.DataFrame({'trackId1': trackId1, 'trackId2': trackId2, 'angle1': angle1,
                       'bearing1': bearing1, 'angle2': angle2, 'bearing2': bearing2,
                       'distance': distances, 'angDistance': angDistances, 'frame': frame,
                       'distFromStart1': distsFromStart1, 'distFromEnd1': distsFromEnd1,
                       'distFromStart2': distsFromStart2, 'distsFromEnd2': distsFromEnd2})

    df.to_pickle('/home/itskov/Dropbox/dist_corr.pkl')





if __name__ == "__main__":
    checkCorrelation('/home/itskov/Temp/behav/04-Mar-2020/TPH1_ATR_TRAIN_75M_D0.avi_16.57.20/exp.npy')
    #checkCorrelation('/home/itskov/Temp/behav/TS1_ATR_TRAIN_75M_0D.avi_11.17.28/exp.npy')
