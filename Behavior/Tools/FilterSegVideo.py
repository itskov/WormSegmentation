import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


from scipy.ndimage import measurements, label
from skvideo.io import FFmpegWriter
from os import path
from time import time

def main(filename):
    cap = cv2.VideoCapture(filename)
    frame_count = np.min((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1000))

    output_filename = path.join(path.dirname(filename), path.basename(filename)[0:-4] + "_flt.mp4")
    full_vid_handle = FFmpegWriter(output_filename)

    for i in range(frame_count):
        _, current_frame = cap.read()
        current_frame = current_frame
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        #labeled_frame, n = label(current_frame)
        labeled_frame = current_frame

        #area = measurements.sum(labeled_frame != 0, labeled_frame, range(n))
        #badAreas = set(np.where((area < 180) | (area > 850))[0])
        before = time()
        #labeled_frame[np.isin(labeled_frame, set(badAreas))] = 0
        #labeled_frame[labeled_frame == badAreas[0]] = 0
        #should_remove = [[(i in badAreas for i in r] for r in labeled_frame]
        labeled_frame[labeled_frame != 0] = 1
        labeled_frame = ndimage.binary_opening(labeled_frame, structure=np.ones((5, 5))).astype(np.int32)
        after = time() - before
        print('Line time: %f' % (after,))
        labeled_frame[labeled_frame > 0] = 255


        full_vid_handle.writeFrame(labeled_frame)

        print('Frame: %d' % (i,))
        pass

    full_vid_handle.close()

if __name__ == "__main__":
    main('/home/itskov/Temp/behav/27-Feb-2020/TPH_1_ATR_TRAIN_75M_0D.avi_11.49.56/27-Feb-2020-11.49.56-MIC2-TPH_1_ATR_TRAIN_75M_0D.avi_seg.mp4')

    47