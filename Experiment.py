import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from Track import Track

from AngleVisualizer import AngleVisualizer
from SegmentedTracker import SegmentedTracker


class Experiment:
    def __init__(self, videoFilename, tracks=None):
        self._tracks = tracks
        self._videoFilename = videoFilename
        self._cap = cv2.VideoCapture(videoFilename)
        self._pointsOfInterest = {}


    def addCirclePosition(self, pointName):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        self._pointsOfInterest[pointName + 'Pos'] = newPoints[0]
        self._pointsOfInterest[pointName + 'Rad'] = newPoints[1]






if __name__ == "__main__":
    rawFile = '/home/itskov/Temp/21.07.19/NOATRNOLIGHT/21-Jul-2019-10.16.35-MIC2-TPH_1_NO_ATR_NO_LIGHT_EXP_DAM4.avi_0_Compressed.mp4'

    tracksDicts = \
        np.load('/home/itskov/Temp/21.07.19/NOATRNOLIGHT/21-Jul-2019-10.16.35-MIC2-TPH_1_NO_ATR_NO_LIGHT_EXP_DAM4.avi_0_Full_seg_tracks.npy')

    # TEMP
    st = SegmentedTracker(rawFile, rawFile)
    st._tracks = tracksDicts
    st.filterTracks()
    tracksDict = st._tracks
    # TEMP


    lens = np.asarray([len(list(t.values())) for t in tracksDicts])
    tracksDicts = tracksDicts[lens > 350]

    tracks = [Track(trackDict) for trackDict in tracksDicts]

    exp = Experiment(rawFile, tracks)
    exp.addCirclePosition('chem')
    av = AngleVisualizer(exp, tracks)
    av.visualize()


    pass;



