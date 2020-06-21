import tensorflow as tf
import numpy as np


from scipy.ndimage import binary_opening
from Behavior.Pipeline.AnalysisStep import AnalysisStep
from trainModel import cnn_model_fn

from os import path
import cv2


class SegmentStep(AnalysisStep):
    def __init__(self):
        self._BINS = 4

    def process(self, artifacts):
        current_frame = artifacts['current_frame']
        restore_point = artifacts['restore_points']


        height, width = current_frame.shape

        if 'sess' not in artifacts:
            currentFrame_ = tf.placeholder(tf.float32, [None, int(height / self._BINS), int(width / self._BINS)])
            filteredFrame_ = tf.placeholder(tf.float32, [None, int(height / self._BINS), int(width / self._BINS)])

            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'

            sess = tf.Session(config=config)
            loss, output = cnn_model_fn(currentFrame_, filteredFrame_,
                                        (int(height / self._BINS), int(width / self._BINS)))

            saver = tf.train.Saver()
            saver.restore(sess, restore_point)

            mj2_path = artifacts['mj2_path']
            input_dir = path.dirname(mj2_path)
            file_Name = path.basename(mj2_path)[0:-4]

            seg_output_filename = path.join(input_dir, file_Name + "_seg.mp4")

            artifacts['seg_vid_filename'] = seg_output_filename
            artifacts['sess'] = sess
            artifacts['sess_output'] = output
            artifacts['sess_current_frame'] = currentFrame_
            artifacts['sess_current_filtered_frame'] = filteredFrame_

        current_frame = np.reshape(artifacts['current_frame'], (1, height, width))
        splitted_frame = self.splitBatch(current_frame, self._BINS)

        procDict = {artifacts['sess_current_frame']: splitted_frame,
                    artifacts['sess_current_filtered_frame']: splitted_frame}

        # The actual forward move.
        output_val = artifacts['sess_output'].eval(procDict, session=artifacts['sess'])
        # Cross entropy
        output_val[..., 0][output_val[..., 0] == 0.5] = 0.49
        output_val = np.argmax(output_val, axis=3)
        #

        output_val = self.mergeBatch(output_val, self._BINS)
        output_val = (np.reshape(output_val, (1, height, width)))

        output_val[0, :, :] = binary_opening(output_val[0, :, :], structure=np.ones((3, 3))).astype(np.int16)
        #output_val[output_val < 100] = 0
        output_val[output_val > 0] = 1 * 255

        artifacts['segmented_frame'] = output_val

        return artifacts







    def close(self, artifacts):
        pass

    def stepName(self, artifacts):
        return 'Segmentation'

    def checkDependancies(self, artifacts):
        if 'current_frame' not in artifacts:
            raise Exception('Cant find current_frame in artifacts.')
        if 'restore_points' not in artifacts:
            raise Exception('Cant find current_frame in artifacts.')

    def splitBatch(self, batchData, bins):
        batchDataSize = batchData.shape

        if (batchDataSize[1] % bins != 0) or (batchDataSize[2] % bins != 0):
            print(batchDataSize)
            print("Error splitting: " + str((batchDataSize[0], batchDataSize[1])) + " do  not divide by " + str(bins))
            return

        #print((batchDataSize[1], batchDataSize[2], batchDataSize[0]))
        # batchData = np.reshape(batchData, (batchDataSize[1],batchDataSize[2],batchDataSize[0]))
        batchDataSize = batchData.shape

        # Splitting the rows.
        rowSplit = np.split(batchData, bins, axis=1)
        # Splitting the cols.
        colSplit = np.asarray([np.asarray(np.split(s, bins, axis=2)) for s in rowSplit])

        # Changing the axis order
        colSplit = np.rollaxis(colSplit, axis=2)

        splittedBatch = np.reshape(np.asarray(colSplit),
                                   (batchDataSize[0] * bins ** 2, int(batchDataSize[1] / bins),
                                    int(batchDataSize[2] / bins)))

        return splittedBatch

    def mergeBatch(self, batchData, bins):

        s = batchData.shape

        reshaedSplittedData = np.reshape(batchData, (int(s[0] / bins ** 2), bins, bins, s[1], s[2]))
        reshaedSplittedData = np.transpose(reshaedSplittedData, (0, 1, 3, 2, 4))

        fullData = np.reshape(reshaedSplittedData, (int(s[0] / bins ** 2), s[1] * bins, s[2] * bins))

        return fullData



