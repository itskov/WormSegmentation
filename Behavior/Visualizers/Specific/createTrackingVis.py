from SegmentedTracker import SegmentedTracker
from Behavior.General.TracksFilter import filterTracksForAnalyses
from Behavior.General.ExpDir import ExpDir

from os import path


import numpy as np


def main(filenmame):
    expDir = ExpDir(expDir=path.dirname(filename))
    exp = np.load(filename)[0]
    tracks = filterTracksForAnalyses(exp._tracks, minSteps=18, minDistance=80)

    tracks = [t.getTrackDict() for t in tracks]

    st = SegmentedTracker(expDir.getExpSegVid(), expDir.getVidFile())
    st._tracks = tracks
    st.createTrackedMovie()




if __name__ == "__main__":
    filename = '/home/itskov/Temp/behav/27-Feb-2020/TPH_1_ATR_TRAIN_75M_0D.avi_14.20.27/exp.npy'
    main(filename)