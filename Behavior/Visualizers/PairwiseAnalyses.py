import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ggplot import ggplot, aes, geom_density, xlab, ylab


from Behavior.Visualizers.RoiAnalysis import RoiAnalysis
from Behavior.Visualizers.OccupVisualizer import OccupVisualizer



def PairWiseRoi(cond1, firstExp, cond2, secondExp):
    sns.set()

    print('Start Analyses..')
    firstRoi = RoiAnalysis(firstExp)
    secondRoi = RoiAnalysis(secondExp)

    firstRoi.execute()
    print('Finished First exp.')
    secondRoi.execute()
    print('Finished Second exp.')

    print(firstRoi._results)
    plt.plot(firstRoi._results['arrivedFrac'], label=cond1)
    plt.plot(secondRoi._results['arrivedFrac'], label=cond2)
    plt.xlabel('Frames (2Hz)')
    plt.ylabel('Worms Arrived')
    plt.legend()
    plt.show()

def PairWiseProjectionDensity(cond1, firstExp, cond2, secondExp):
    LENGTH_THR = 250

    sns.set()

    print('Start Analyses..')

    firstProj = [track.getMeanProjection(firstExp._regionsOfInterest['endReg']['pos']) for
                 track in firstExp._tracks if track._trackCords.shape[0] >= LENGTH_THR]
    secondProj = [track.getMeanProjection(secondExp._regionsOfInterest['endReg']['pos']) for
                  track in secondExp._tracks if track._trackCords.shape[0] >= LENGTH_THR]

    firstDf = pd.DataFrame({'proj' : firstProj, 'cond' : cond1})
    secondDf = pd.DataFrame({'proj': secondProj, 'cond': cond2})


    allProj = list(firstDf['proj'].values) + list(secondDf['proj'].values)
    allConds = list(firstDf['cond'].values) + list(secondDf['cond'].values)
    df = pd.DataFrame({'proj' : allProj, 'cond' : allConds})


    #df['cond'] = df['cond'].astype('category')
    g = ggplot(aes(x='proj', color='cond'), data=df) + geom_density(alpha=1) + xlab('Projection') + ylab('Density')

    print(g)
    #g.draw()

def PairWiseOccupVisoulatizer(cond1, firstExp, cond2, secondExp):
    firstOccup = OccupVisualizer(firstExp)
    secondOccup = OccupVisualizer(secondExp)

    firstOccup.execute(cond1)
    secondOccup.execute(cond2)
