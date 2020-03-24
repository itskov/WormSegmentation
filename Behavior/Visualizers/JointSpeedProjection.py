import seaborn as sns
import pandas as pd

from Behavior.General.TracksFilter import filterTracksForAnalyses


class JointSpeedProjection:
    def __init__(self, exp):
        self._exp = exp
        self._tracks = filterTracksForAnalyses(exp._tracks, minSteps=50, minDistance=50)

    def execute(self):
        speeds = [t.getMeanSpeed() for t in self._tracks]
        projs = [t.getMeanProjection(self._exp._regionsOfInterest['endReg']['pos'])]

        df = pd.DataFrame({'speed': speeds, 'proj': projs})
        plt.style.use('dark_background')
        sns.jointplot(x='speed', y='proj', data=df, kind='scatter')
