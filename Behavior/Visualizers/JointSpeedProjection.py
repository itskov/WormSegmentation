import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from Behavior.General.TracksFilter import filterTracksForAnalyses


class JointSpeedProjection:
    def __init__(self, exp):
        self._exp = exp
        self._tracks = filterTracksForAnalyses(exp._tracks, minSteps=50, minDistance=50)

        self._speeds = [t.getMeanSpeed() for t in self._tracks]
        self._projs = [t.getMeanProjection(self._exp._regionsOfInterest['endReg']['pos']) for t in self._tracks]


    def execute(self, xlims, ylims):
        plt.style.use('deafault')

        df = pd.DataFrame({'speed': self._speeds, 'proj': self._projs})
        sns.jointplot(x='speed', y='proj', data=df, kind='scatter', xlim=xlims, ylim=ylims)

        return(self._speeds, self._projs)
