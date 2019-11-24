import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

class RevPosDensity:
    def __init__(self, exp):
        self._exp = exp
        self._tracks = exp._tracks

    def execute(self, condName, pyplotPlot = False):
            df = pd.DataFrame(columns=['reversalsDistances'])
            for i, t in enumerate(self._tracks):
                if t._trackCords.shape[0] < 500:
                    continue
                if t.getMaxDistTravelled() < 300:
                    continue

                tracksReversals = t._tracksReversals
                distances = t.getDistances(self._exp._regionsOfInterest['endReg']['pos'])

                revDistances = distances[tracksReversals]

                currentDf = pd.DataFrame({'reversalsDistances': revDistances})
                df = pd.concat([df, currentDf])

                #print('Track %d' % i)

            #sns.jointplot(x=df["dist"], y=df["ang"], kind='kde')
            ax = sns.kdeplot(df['reversalsDistances'], shade=True, label=condName)
            ax.set(xlabel="Reversal Distance", ylabel="Density")

            if pyplotPlot:
                plt.show()

if __name__ == "__main__":
    EXP_PATH = '/mnt/storageNASRe/ChristianData/ChrisNewTracks/01-Nov-2018_Chris/LTAV.avi_15.26.05/exp.npy'
    exp = np.load(EXP_PATH)[0]
    rd = RevPosDensity(exp)
    rd.execute()