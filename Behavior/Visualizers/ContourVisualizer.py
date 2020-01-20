import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns

class ContourVisualizer():
    def __init__(self, exp):
        self._exp = exp
        self._frameSize = exp.getFrameSize()
        self._results = {}

    def execute(self, title="", size=None, showPlot=True):
        allCords = np.array([])
        for i, track in enumerate(self._exp._tracks):
            #print('Track %d' % i)
            if track.getMaxDistTravelled() > 50:
                cords = track._trackCords.astype(np.int)
                allCords = np.vstack((allCords, cords)) if allCords.size else cords

        # Saving the results
        self._results['allCords'] = allCords

        if showPlot == True:
            #plt.style.use("dark_background")
            plt.style.use("seaborn-darkgrid")
            my_cmap = ListedColormap(sns.dark_palette("purple").as_hex())

            sns.kdeplot(allCords[:, 0], allCords[:, 1], cmap=my_cmap, shade=True)
            plt.axis('off')
            plt.title(title)
            plt.show()

        pass

if __name__ == "__main__":
    exp = np.load('/home/itskov/Temp/behav/15-Jan-2020/TPH_1_ATR_TRAIN_60M_D120_NO_IAA3x5.avi_17.27.06/exp.npy')[0]
    vis = ContourVisualizer(exp)
    vis.execute()



