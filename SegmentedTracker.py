import cv2
import os
import sys

import numpy as np

from scipy.ndimage import label
from scipy.spatial.distance import  pdist
from skvideo.io import FFmpegWriter

from PIL import Image, ImageDraw, ImageFont

class SegmentedTracker:
    def __init__(self, segmentedFile):
        self._inputFile = segmentedFile

        self._path = os.path.dirname(segmentedFile)
        self._baseName = os.path.basename(segmentedFile)

        self._cap = cap = cv2.VideoCapture(self._inputFile)
        #self._numOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._numOfFrames = 250

        self._tracks = []

    def track(self):
        # Going to the first frame.
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

        currentTracks = []

        for currentFrameNum in range(self._numOfFrames):
            readFrame, labeledFrame, n = self.getFrame()
            shouldKeepTracks = np.ones((len(currentTracks),), dtype=np.bool)

            # Prepare centroids
            centroids = np.zeros((n, 2), dtype=np.int)
            usedCentroids = np.zeros((n, 1))
            for li, l in enumerate(np.unique(labeledFrame)):
                if (l == 0):
                    continue

                x, y = np.where(labeledFrame == l)
                centroids[li, :] = np.array((int(np.mean(x)), int(np.mean(y))))
                usedCentroids[li] = 0

            if (currentFrameNum > 0):
                for ti, t in enumerate(currentTracks):
                    distances = []
                    if (currentFrameNum - 1) in t:
                        distances = [np.linalg.norm(cent - t[currentFrameNum - 1]) for cent in centroids]
                    elif (currentFrameNum - 2) in t:
                        distances = [np.linalg.norm(cent - t[currentFrameNum - 2]) for cent in centroids]
                    else:
                        shouldKeepTracks[ti] = False

                    if (len(distances) > 0):
                        nextPosIndex = np.argmin(distances)
                        if (usedCentroids[nextPosIndex] == 0 and distances[nextPosIndex] < 20):
                            t[currentFrameNum] = centroids[np.argmin(distances),:]
                            usedCentroids[nextPosIndex] = 1


            #
            if (shouldKeepTracks.size > 0):
                self._tracks += list(np.asanyarray(currentTracks)[np.logical_not(shouldKeepTracks)])
                currentTracks = list(np.asanyarray(currentTracks)[shouldKeepTracks])

            # Adding unmatched centroids as tracks.
            [currentTracks.append({currentFrameNum: cent}) for cent in centroids[np.ravel(usedCentroids) == 0, :]]

            # Log
            print('Tracking frame: ' + str(currentFrameNum) + " Entites in frame: " + str(n))

        self._tracks += list(currentTracks)


    def filterTracks(self):
        lens = np.asarray([len(list(t.values())) for t in self._tracks])
        self._tracks = np.asarray(self._tracks)[lens > 5]

        maxDistances = [max(pdist(np.asarray(t.values))) for t in self._tracks]


    def createTrackedMovie(self):
        # Going to the first frame.
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

        outputFile = os.path.join(self._path,self._baseName,'_tracked.mp4')
        videoWriter = FFmpegWriter(outputFile, outputdict={'-crf': '0'})

        font = ImageFont.truetype("FreeSans.ttf", 32)

        for currentFrameNum in range(1, self._numOfFrames):
            print('Saving frame: ' + str(currentFrameNum))
            readFrame, _,_n = self.getFrame()

            curIm = Image.fromarray(readFrame).convert('RGB')
            curImDraw = ImageDraw.Draw(curIm)

            for t in self._tracks:
                if currentFrameNum in t:
                    trajItems = list(t.items())

                    traj = [(pos[1][1], pos[1][0]) for pos in trajItems if pos[0] <= currentFrameNum]

                    curImDraw.line(traj, fill=(255,0,0), width=2)
                    curImDraw.text(traj[-1], "+", (0, 0, 255), font=font)

            videoWriter.writeFrame(np.asarray(curIm))

        videoWriter.close()





    def getFrame(self):
        success, readFrame = self._cap.read()
        readFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2GRAY)
        labeledFrame, n = label(np.uint16(readFrame))

        for j in range(n):
            if (np.sum(labeledFrame == j) < 25 or np.sum(labeledFrame == j) > 300):
                labeledFrame[labeledFrame == j] = 0


        n = len(np.unique(labeledFrame))
        return (readFrame, labeledFrame, n)


if __name__ == "__main__":
    tracker = SegmentedTracker(sys.argv[1])
    tracker.track()
    tracker.filterTracks()
    tracker.createTrackedMovie()
