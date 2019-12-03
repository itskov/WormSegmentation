import numpy as np
import matplotlib.pyplot as plt

from cv2 import blur

class OccupVisualizer():
    def __init__(self, exp):
        self._exp = exp
        self._frameSize = exp.getFrameSize()

    def execute(self, title="", size=None):
        field = np.zeros(self._frameSize)

        for i, track in enumerate(tracks):
            #print('Track %d' % i)
            if track.getMaxDistTravelled() > 300:
                cords = track._trackCords.astype(np.int)
                field[cords[:,0], cords[:,1]] += 1

        # First we roll the image to center the plate.
        rightBorder = np.min(np.where(field > 0)[1])
        leftBorder = field.shape[1] - np.max(np.where(field > 0)[1])
        allBorders = rightBorder + leftBorder
        correctBorder = np.floor(allBorders / 2)
        field = np.roll(field, int(correctBorder - rightBorder), axis=1)

        nField = (field - np.min(field)) / (np.max(field) - np.min(field))
        nField = blur(nField,(48,48))
        plt.imshow(nField, cmap=plt.get_cmap('gnuplot2'))
        plt.title(title)
        plt.axis('off')
        plt.show()



if __name__ == "__main__":
    exp = np.load('/home/itskov/Temp/behav/28-Nov-2019/TPH_1_NO_ATR_TRAIN_NO_IAA3x5.avi_11.58.42/exp.npy')[0]
    vis = OccupVisualizer(exp)
    vis.execute()



