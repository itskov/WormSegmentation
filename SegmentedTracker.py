import cv2
import os
import sys


import numpy as np

from scipy.ndimage import measurements, label
from scipy.spatial.distance import  pdist
from skvideo.io import FFmpegWriter
from time import time

from PIL import Image, ImageDraw, ImageFont

#from tensorflow.contrib.image import connected_components
from tensorflow import Session

class SegmentedTracker:
    def __init__(self, segmentedFile, rawInputFile):
        self._segmentedInputFile = segmentedFile
        self._rawInputFile = rawInputFile

        self._path = os.path.dirname(segmentedFile)
        self._baseName = os.path.basename(segmentedFile)

        # Getting rid of the extension
        self._baseName = self._baseName[0:-4]

        self._segmentedCap = cv2.VideoCapture(self._segmentedInputFile)
        self._rawCap = cv2.VideoCapture(self._rawInputFile)

        self._numOfFrames = int(self._segmentedCap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
        # self._numOfFrames = 350

        self._tracks = []

        self._session = Session()

    def track(self):
        # Calculating the mean intensity.
        #self._segmentedCap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        #_, rawReadFrame,_, _ = self.getFrame(False)
        #initialMeanIntensity = np.mean(rawReadFrame)

        # Going to the first frame.
        self._segmentedCap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        currentTracks = []

        #for currentFrameNum in range(self._numOfFrames):
        for currentFrameNum in range(100):
            startTime = time()	
            readFrame, rawReadFrame, labeledFrame, labelsInds = self.getFrame()
            shouldKeepTracks = np.ones((len(currentTracks),), dtype=np.bool)

            # Prepare centroids
            centroids = np.zeros((len(labelsInds), 2), dtype=np.int)
            usedCentroids = np.zeros((len(labelsInds), 1))
            #for li, l in enumerate(labelsInds):
            #    if (l == 0):
            #        continue

                #if (np.mean(rawReadFrame) > 20 * initialMeanIntensity):
                #    continue;

            #    x, y = np.where(labeledFrame == l)
            #    centroids[li, :] = np.array((int(np.mean(x)), int(np.mean(y))))
            #    usedCentroids[li] = 0
            centroids = measurements.center_of_mass(labeledFrame, labeledFrame)
            centroids = [np.array(cent) for cent in centroids]
            print(centroids)



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
                        if (usedCentroids[nextPosIndex] == 0 and distances[nextPosIndex] < 18):
                            t[currentFrameNum] = centroids[np.argmin(distances),:]
                            usedCentroids[nextPosIndex] = 1


            #
            if (shouldKeepTracks.size > 0):
                self._tracks += list(np.asanyarray(currentTracks)[np.logical_not(shouldKeepTracks)])
                currentTracks = list(np.asanyarray(currentTracks)[shouldKeepTracks])

            # Adding unmatched centroids as tracks.
            [currentTracks.append({currentFrameNum: cent}) for cent in centroids[np.ravel(usedCentroids) == 0, :]]

            # Log
            print('Tracking frame: ' + str(currentFrameNum) + " Entites in frame: " + str(len(labelsInds)) + ". Time: " + str(time() - startTime))

        self._tracks += list(currentTracks)


    def filterTracks(self):
        lens = np.asarray([len(list(t.values())) for t in self._tracks])
        self._tracks = np.asarray(self._tracks)[lens > 5]

        maxDistances = [max(pdist(np.asarray(list(t.values())))) for t in self._tracks]
        self._tracks = self._tracks[np.asarray(maxDistances) > 50]



    def createTrackedMovie(self):
        # Going to the first frame.
        self._segmentedCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._rawCap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        outputFileSeg = os.path.join(self._path,self._baseName +'_seg_tracked.mp4')
        outputFileRaw = os.path.join(self._path,self._baseName +'_raw_tracked.mp4')
        outputFileBoth = os.path.join(self._path,self._baseName +'_both_tracked.mp4')

        print(outputFileSeg)
        print(outputFileRaw)
        print(outputFileBoth)

        videoWriterSeg = FFmpegWriter(outputFileSeg, outputdict={'-crf': '0'})
        videoWriterRaw = FFmpegWriter(outputFileRaw, outputdict={'-crf': '20'})
        videoWriterBoth = FFmpegWriter(outputFileBoth, outputdict={'-crf': '20'})

        font = ImageFont.truetype("FreeSans.ttf", 32)

        for currentFrameNum in range(1, self._numOfFrames):
            print('Saving frame: ' + str(currentFrameNum))
            segReadFrame, rawReadFrame,_,_ = self.getFrame(False)

            # The segmented output
            curImSeg = Image.fromarray(segReadFrame).convert('RGB')
            curImSegDraw = ImageDraw.Draw(curImSeg)

            # The raw output
            curImRaw = Image.fromarray(rawReadFrame).convert('RGB')
            curImRawDraw = ImageDraw.Draw(curImRaw)

            for t in self._tracks:
                if currentFrameNum in t:
                    trajItems = list(t.items())

                    traj = [(pos[1][1], pos[1][0]) for pos in trajItems if pos[0] <= currentFrameNum]

                    #curImSegDraw.line(traj, fill=(255,0,0), width=2)
                    #curImSegDraw.text(traj[-1], "+", (0, 0, 255), font=font)

                    curImRawDraw.line(traj, fill=(255,0,0), width=2)
                    curImRawDraw.text(traj[-1], "+", (0, 0, 255), font=font)


            videoWriterSeg.writeFrame(np.asarray(curImSeg).copy())
            videoWriterRaw.writeFrame(np.asarray(curImRaw).copy())

            bothFrame = np.concatenate((np.asarray(curImSeg).copy(), np.asarray(curImRaw).copy()), axis=1)
            videoWriterBoth.writeFrame(bothFrame)

        videoWriterSeg.close()
        videoWriterRaw.close()
        videoWriterBoth.close()





    def getFrame(self, shouldLabel=True):
        success, readFrame = self._segmentedCap.read()

        segReadFrame = cv2.cvtColor(readFrame, cv2.COLOR_BGR2GRAY)

        if (shouldLabel):
            #labeledFrame = connected_components(np.uint16(segReadFrame))
            #labeledFrame = labeledFrame.eval(session = self._session)
            labeledFrame, n = label(np.uint16(segReadFrame))

            n = len(np.unique(labeledFrame))
            initialLabelsInds =  list(range(n))

            area = measurements.sum(labeledFrame != 0, labeledFrame, index=list(range(n)))
            badAreas = np.where((area < 25) | (area > 400))[0]
            labeledFrame[np.isin(labeledFrame, badAreas)] = 0

            labelsInds = set(list(initialLabelsInds)).difference(set(list(badAreas)))
        else:
            labeledFrame = segReadFrame
            labelsInds = []

        success, rawReadFrame = self._rawCap.read()

        return (segReadFrame, rawReadFrame, labeledFrame, labelsInds)


    def saveTracks(self):
        outputFileBoth = os.path.join(self._path, self._baseName + '_tracks')
        np.save(outputFileBoth, self._tracks)


if __name__ == "__main__":
    tracker = SegmentedTracker(sys.argv[1], sys.argv[2])
    #tracker = SegmentedTracker('/home/itskov/Temp/outputFile.mp4','/home/itskov/Temp/outputFile.mp4')
    tracker.track()
    tracker.filterTracks()
    tracker.createTrackedMovie()

    tracker.saveTracks()
