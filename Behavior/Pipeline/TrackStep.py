from Behavior.Pipeline.AnalysisStep import AnalysisStep
from Behavior.General.Track import Track
from scipy.ndimage import measurements, label
from scipy.spatial.distance import pdist
from scipy import ndimage
from os import path

import numpy as np

#import tensorflow as tf



class TrackStep(AnalysisStep):

    def __init__(self):
        self._tracks = []
        self._currentTracks = []


        #self._sess = tf.Session()
        #self._sess.run(tf.global_variables_initializer())

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
                    #cur_mat = np.array(centroids) - np.array(t[frame_num - 1])
                    #distances = self._sess.run(tf.norm(tf.convert_to_tensor(cur_mat), axis=1))
                elif (frame_num - 2) in t:
                    distances = np.linalg.norm(np.array(centroids) - np.array(t[frame_num - 2]), axis=1)
                    #cur_mat = np.array(centroids) - np.array(t[frame_num - 2])
                    #distances = self._sess.run(tf.norm(tf.convert_to_tensor(cur_mat), axis=1))
                else:
                    shouldKeepTracks[ti] = False

                if len(distances) > 0:
                    nextPosIndex = np.argmin(distances)
                    if (usedCentroids[nextPosIndex] == 0 and distances[nextPosIndex] < 25):
                        t[frame_num] = centroids[np.argmin(distances), :]
                        usedCentroids[nextPosIndex] = 1

        if shouldKeepTracks.size > 0:
            self._tracks += list(np.asanyarray(self._currentTracks)[np.logical_not(shouldKeepTracks)])
            self._currentTracks = list(np.asanyarray(self._currentTracks)[shouldKeepTracks])

        # Adding unmatched centroids as tracks.
        if centroids.size > 0:
            [self._currentTracks.append({frame_num: cent}) for cent in centroids[np.ravel(usedCentroids) == 0, :]]

        # Log
        # print('Tracking frame: ' + str(frame_num) + " Entities in frame: " + str(len(labelsInds)))

        return artifacts


    def filterTracks(self):
        # Then filter them.
        lens = np.asarray([len(list(t.values())) for t in self._tracks])
        print('Filtering tracks..')
        print('Before filtering: ' + str(len(self._tracks)) + " tracks.")
        self._tracks = np.asarray(self._tracks)[lens > 5]
        maxDistances = [max(pdist(np.asarray(list(t.values())))) for t in self._tracks]
        self._tracks = self._tracks[np.asarray(maxDistances) > 7]
        print('After filtering by length: ' + str(self._tracks.shape) + " tracks.")


    def close(self, artifacts):
        #self._sess.close()

        self._tracks += list(self._currentTracks)

        # Fix frames indices
        self.filterTracks()
        self._tracks = [self.orderTrack(track) for track in self._tracks]

        # Writing the tracks files
        mj2_path = artifacts['mj2_path']
        inputPath = path.dirname(mj2_path)
        baseName = ".".join(path.basename(mj2_path).split(".")[0:-1])

        outputFile = path.join(inputPath, baseName + "_tracks")

        tracks = [Track(t) for t in self._tracks]

        np.save(outputFile, tracks)

        # Temp. Create the tracking files.
        from SegmentedTracker import SegmentedTracker
        st = SegmentedTracker(artifacts['seg_vid_filename'], artifacts['full_vid_filename'])
        st._tracks = self._tracks
        st.createTrackedMovie()


    def stepName(self, artifacts):
        return 'Tracking'

    def checkDependancies(self, artifacts):
        if 'segmented_frame' not in artifacts:
            raise Exception('Cant find segmented_frame in artifacts.')


    def orderTrack(self, track):
        dictItems = list(track.items())
        [frames, poses] = list(zip(*dictItems))
        sortIndices = np.argsort(frames)

        # Sorted frames
        frames = list(np.array(frames)[sortIndices])
        poses = np.array(poses)[sortIndices]

        pairs = list(zip(frames, poses))
        track = dict(pairs)
        return track



    def getFrame(self, segFrame, shouldLabel=True):
        #segReadFrame = cv2.cvtColor(segFrame, cv2.COLOR_BGR2GRAY)
        segReadFrame = segFrame

        if (shouldLabel):

            #labeledFrame = np.squeeze(labeledFrame)

            #n = len(np.unique(labeledFrame))


            #area = measurements.sum(labeledFrame != 0, labeledFrame, index=list(range(n)))
            #badAreas = (np.where((area < 8) | (area > 400))[0])
            #labeledFrame[np.isin(labeledFrame, badAreas)] = 0

            #eraseFunc = lambda p: 0 if p in badAreas else p
            #labeledFrame = np.array([eraseFunc(p) for p in np.ravel(labeledFrame)])
            segFrame[segFrame != 0] = 1
            print(segFrame.shape)
            filtered_frame = ndimage.binary_opening(segFrame, structure=np.ones((1, 4, 4))).astype(np.int16)
            labeledFrame, n = label(np.uint16(filtered_frame))
            labelsInds = set(range(n))

            #labelsInds = initialLabelsInds.difference(badAreas)
        else:
            filtered_frame = segReadFrame
            labelsInds = []

        return segReadFrame, filtered_frame, labelsInds



