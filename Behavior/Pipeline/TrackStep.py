from Behavior.Pipeline.AnalysisStep import AnalysisStep
from scipy.ndimage import measurements, label
from os import path

import numpy as np

import cv2

class TrackStep(AnalysisStep):

    def __init__(self):
        self._tracks = []
        self._currentTracks = []

    # Return None if failed.
    def process(self, artifacts):
        frame_num = artifacts['frame_num']
        readFrame, labeledFrame, labelsInds = self.getFrame(artifacts['segmented_frame'])
        shouldKeepTracks = np.ones((len(self._currentTracks),), dtype=np.bool)

        # Prepare centroids
        usedCentroids = np.zeros((len(labelsInds), 1), dtype=np.bool)

        centroids = measurements.center_of_mass(labeledFrame, labels=labeledFrame, index=np.unique(labeledFrame))
        centroids = centroids[1:]
        centroids = np.asarray([np.array(cent) for cent in centroids])

        if frame_num > 0 and centroids.size > 0:
            for ti, t in enumerate(self._currentTracks):
                distances = []
                if (frame_num - 1) in t:
                    distances = np.linalg.norm(np.array(centroids) - np.array(t[frame_num - 1]), axis=1)
                elif (frame_num - 2) in t:
                    distances = np.linalg.norm(np.array(centroids) - np.array(t[frame_num - 2]), axis=1)
                else:
                    shouldKeepTracks[ti] = False

                if (len(distances) > 0):
                    nextPosIndex = np.argmin(distances)
                    if (usedCentroids[nextPosIndex] == 0 and distances[nextPosIndex] < 25):
                        t[frame_num] = centroids[np.argmin(distances), :]
                        usedCentroids[nextPosIndex] = 1

        if (shouldKeepTracks.size > 0):
            self._tracks += list(np.asanyarray(self._currentTracks)[np.logical_not(shouldKeepTracks)])
            currentTracks = list(np.asanyarray(self._currentTracks)[shouldKeepTracks])

        # Adding unmatched centroids as tracks.
        if centroids.size > 0:
            [currentTracks.append({frame_num: cent}) for cent in centroids[np.ravel(usedCentroids) == 0, :]]

        # Log
        print('Tracking frame: ' + str(frame_num) + " Entities in frame: " + str(len(labelsInds)))



    def close(self, artifacts):
        self._tracks += list(self._currentTracks)

        # Fix frames indices
        self._tracks = [self.orderTrack(track) for track in self._tracks]

        # Writing the tracks files
        mj2_path = artifacts['mj2_path']
        inputPath = path.dirname(mj2_path)
        baseName = ".".join(path.basename(mj2_path).split(".")[0:-1])

        outputFile = path.join(inputPath, baseName + "_tracks")
        np.save(outputFile, self._tracks)

        # Now create the movies.
        from SegmentedTracker import SegmentedTracker



    def stepName(self, artifacts):
        return 'Tracking'

    def checkDependancies(self, artifacts):
        if 'segmented_frame' not in artifacts:
            raise Exception('Cant find segmented_frame in artifacts.')


    def getFrame(self, segFrame, shouldLabel=True):
        #segReadFrame = cv2.cvtColor(segFrame, cv2.COLOR_BGR2GRAY)
        segReadFrame = segFrame

        if (shouldLabel):
            labeledFrame, n = label(np.uint16(segReadFrame))

            n = len(np.unique(labeledFrame))
            initialLabelsInds =  list(range(n))

            area = measurements.sum(labeledFrame != 0, labeledFrame, index=list(range(n)))
            badAreas = np.where((area < 5) | (area > 400))[0]
            labeledFrame[np.isin(labeledFrame, badAreas)] = 0

            labelsInds = set(list(initialLabelsInds)).difference(set(list(badAreas)))
        else:
            labeledFrame = segReadFrame
            labelsInds = []

        return segReadFrame, labeledFrame, labelsInds



