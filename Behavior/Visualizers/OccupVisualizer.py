import numpy as np
import matplotlib.pyplot as plt

from cv2 import blur, GaussianBlur, BORDER_DEFAULT

class OccupVisualizer():
    def __init__(self, exp):
        self._exp = exp
        self._frameSize = exp.getFrameSize()
        self._results = {}

    def execute(self, title="", size=None, showPlot=True):
        field = np.zeros(self._frameSize)


        #allCords = np.array([])
        for i, track in enumerate(self._exp._tracks):
            #print('Track %d' % i)
            if track.getMaxDistTravelled() > 100:
                cords = track._trackCords.astype(np.int)
                field[cords[:, 0], cords[:, 1]] += 1
                #allCords = np.vstack((allCords, cords)) if allCords.size else cords

        # First we roll the image to center the plate.
        rightBorder = np.min(np.where(field > 0)[1])
        leftBorder = field.shape[1] - np.max(np.where(field > 0)[1])
        allBorders = rightBorder + leftBorder
        correctBorder = np.floor(allBorders / 2)
        field = np.roll(field, int(correctBorder - rightBorder), axis=1)
        nField = (field - np.min(field)) / (np.max(field) - np.min(field))
        #DEBUG
        nField = (np.log10(nField))
        nField[nField == -np.inf] = 0
        #DEBUG
        #nField = blur(nField, (24, 24))
        nField = (GaussianBlur(nField,(65,65),BORDER_DEFAULT,15,15))


        # Saving the results
        self._results['mat'] = nField

        if showPlot == True:
            plt.style.use("dark_background")
            plt.imshow(nField, cmap=plt.get_cmap('gnuplot2_r'))
            plt.title(title)
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    exp = np.load('/home/itskov/Temp/behav/15-Jan-2020/TPH_1_ATR_TRAIN_60M_D120_NO_IAA3x5.avi_17.27.06/exp.npy')[0]
    vis = OccupVisualizer(exp)
    vis.execute()



