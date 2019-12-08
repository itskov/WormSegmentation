import pandas as pd
import seaborn as sns


class AngPosDensity:
    def __init__(self, exp):
        self._exp = exp
        self._tracks = exp._tracks
        self._results = {}

    def execute(self):
            df = pd.DataFrame(columns=['dist','ang'])
            for i, t in enumerate(self._tracks):
                if t._trackCords.shape[0] < 500:
                    continue
                if t.getMaxDistTravelled() < 300:
                    continue


                angs = t.getAngles(self._exp._regionsOfInterest['endReg']['pos'])
                distances = t.getDistances(self._exp._regionsOfInterest['endReg']['pos'])

                angs = angs[2:-2]
                distances = distances[2:-2]

                currentDf = pd.DataFrame({'dist': distances, 'ang': angs})
                df = pd.concat([df, currentDf])
                print('Track %d' % i)

            sns.jointplot(x=df["dist"], y=df["ang"], kind='kde')
