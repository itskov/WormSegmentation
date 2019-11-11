import numpy as np
import matplotlib.pyplot as plt

from cv2 import blur

class OccupVisualizer():
    def __init__(self, exp):
        self._exp = exp
        self._frameSize = exp.getFrameSize()

    def execute(self, title=""):
        field = np.zeros(self._frameSize)

        for i, track in enumerate(self._exp._tracks):
            #print('Track %d' % i)
            if track.getMaxDistTravelled() > 300:
                cords = track._trackCords.astype(np.int)
                field[cords[:,0], cords[:,1]] += 1

        nField = (field - np.min(field)) / (np.max(field) - np.min(field))
        nField = blur(nField,(48,48))
        plt.imshow(nField, cmap=plt.get_cmap('gnuplot2'))
        plt.title(title)
        plt.show()



if __name__ == "__main__":
    exp = np.load('/home/itskov/Temp/05-Sep-2019/TPH_1_ATR_TRAIN_NO_IAA3.avi_20.48.41/exp.npy')[0]
    vis = OccupVisualizer(exp)
    vis.execute()



