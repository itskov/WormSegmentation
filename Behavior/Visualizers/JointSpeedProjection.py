import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from Behavior.General.TracksFilter import filterTracksForAnalyses


class JointSpeedProjection:
    def __init__(self, exp):
        self._exp = exp
        self._tracks = filterTracksForAnalyses(exp._tracks, minSteps=50, minDistance=50)

    def execute(self):
        plt.style.use('dark_background')
        speeds = [t.getMeanSpeed() for t in self._tracks]
        projs = [t.getMeanProjection(self._exp._regionsOfInterest['endReg']['pos']) for t in self._tracks]

        df = pd.DataFrame({'speed': speeds, 'proj': projs})
        sns.jointplot(x='speed', y='proj', data=df, kind='scatter')
