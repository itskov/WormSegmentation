import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ggplot import ggplot, aes, geom_density, xlab, ylab


from Behavior.Visualizers.RoiAnalysis import RoiAnalysis
from Behavior.Visualizers.OccupVisualizer import OccupVisualizer
from Behavior.Visualizers.RevPosDensity import RevPosDensity



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
    print(secondRoi._results)
    plt.plot(firstRoi._results['arrivedFrac'], label=" %s, %d worms" % (cond1, firstRoi._results['wormCount']))
    plt.plot(secondRoi._results['arrivedFrac'], label=" %s, %d worms" % (cond2, secondRoi._results['wormCount']))
    plt.xlabel('Frames (2Hz)')
    plt.ylabel('Worms Arrived')
    plt.legend()
    plt.show()

def PairWiseProjectionDensity(cond1, firstExp, cond2, secondExp):
    LENGTH_THR = 250


    print('Start Analyses..')

    firstProj = [track.getMeanProjection(firstExp._regionsOfInterest['endReg']['pos']) for
                 track in firstExp._tracks if track._trackCords.shape[0] >= LENGTH_THR and track.getMaxDistTravelled() > 350]
    secondProj = [track.getMeanProjection(secondExp._regionsOfInterest['endReg']['pos']) for
                  track in secondExp._tracks if track._trackCords.shape[0] >= LENGTH_THR and track.getMaxDistTravelled() > 350]

    firstDf = pd.DataFrame({'proj' : firstProj, 'cond' : cond1})
    secondDf = pd.DataFrame({'proj': secondProj, 'cond': cond2})


    allProj = list(firstDf['proj'].values) + list(secondDf['proj'].values)
    allConds = list(firstDf['cond'].values) + list(secondDf['cond'].values)
    df = pd.DataFrame({'proj' : allProj, 'cond' : allConds})


    sns.kdeplot(firstDf['proj'], shade=True, label=cond1)
    ax = sns.kdeplot(secondDf['proj'], shade=True, label=cond2)

    ax.set(xlabel="Projection", ylabel="Density")
    plt.show()
    #g.draw()

def PairWiseSpeedDensity(cond1, firstExp, cond2, secondExp):
    LENGTH_THR = 250

    sns.set()

    print('Start Analyses..')

    firstProj = [track.getMeanSpeed() / firstExp._scale for
                 track in firstExp._tracks if track._trackCords.shape[0] >= LENGTH_THR]
    secondProj = [track.getMeanSpeed() / firstExp._scale for
                  track in secondExp._tracks if track._trackCords.shape[0] >= LENGTH_THR]

    firstDf = pd.DataFrame({'proj' : firstProj, 'cond' : cond1})
    secondDf = pd.DataFrame({'proj': secondProj, 'cond': cond2})


    allProj = list(firstDf['proj'].values) + list(secondDf['proj'].values)
    allConds = list(firstDf['cond'].values) + list(secondDf['cond'].values)
    df = pd.DataFrame({'proj' : allProj, 'cond' : allConds})


    #df['cond'] = df['cond'].astype('category')
    #g = ggplot(aes(x='proj', color='cond'), data=df) + geom_density(alpha=1) + xlab('Speed') + ylab('Density')
    sns.kdeplot(firstDf['proj'], shade=True, label=cond1)
    ax = sns.kdeplot(secondDf['proj'], shade=True, label=cond2)

    ax.set(xlabel="Velocity", ylabel="Density")
    plt.show()
    #print(g)
    #g.draw()


def PairWiseOccupVisoulatizer(cond1, firstExp, cond2, secondExp):
    firstOccup = OccupVisualizer(firstExp)
    secondOccup = OccupVisualizer(secondExp)

    firstOccup.execute(cond1)
    secondOccup.execute(cond2)

def PairWiseRevDistances(cond1, firstExp, cond2, secondExp):
    rd1 = RevPosDensity(firstExp)
    rd1.execute(cond1)
    rd2 = RevPosDensity(secondExp)
    rd2.execute(cond2)
    plt.show()

