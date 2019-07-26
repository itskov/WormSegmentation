import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from Track import Track

from AngleVisualizer import AngleVisualizer



class Experiment:
    def __init__(self, videoFilename, tracks=None):
        self._tracks = tracks
        self._videoFilename = videoFilename
        self._cap = cv2.VideoCapture(videoFilename)
        self._pointsOfInterest = {}


    def addCirclePosition(self, pointName):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, sampleFrame = self._cap.read()
        plt.imshow(sampleFrame); plt.show()
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
    tracksDicts = np.load('/home/itskov/Temp/tracks2.npy')

    lens = np.asarray([len(list(t.values())) for t in tracksDicts])
    tracksDicts = tracksDicts[lens > 350][1:10]

    tracks = [Track(trackDict) for trackDict in tracksDicts]

    exp = Experiment('/home/itskov/Temp/outputFile.mp4', tracks)
    exp.addCirclePosition('chem')
    av = AngleVisualizer(exp, tracks[1:5])
    av.visualize()


    pass;



