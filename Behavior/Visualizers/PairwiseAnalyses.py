import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ggplot import ggplot, aes, geom_density, xlab, ylab


from Behavior.Visualizers.RoiAnalysis import RoiAnalysis
from Behavior.Visualizers.OccupVisualizer import OccupVisualizer
from Behavior.Visualizers.RevPosDensity import RevPosDensity
from Behavior.Visualizers.JointSpeedProjection import JointSpeedProjection


def PairWiseJointSpeedProjection(cond1, firstExp, cond2, secondExp, showShow=True):
    plt.style.use("dark_background")
    sns.set_context("talk")

    firstJoint = JointSpeedProjection(firstExp)
    secondJoint = JointSpeedProjection(secondExp)

    minSpeed = np.min((np.min(firstJoint._speeds), np.min(secondJoint._speeds)))
    maxSpeed = np.max((np.max(firstJoint._speeds), np.max(secondJoint._speeds)))

    minProj = np.min((np.min(firstJoint._projs), np.min(secondJoint._projs)))
    maxProj = np.max((np.max(firstJoint._projs), np.max(secondJoint._projs)))

    #firstJoint.execute(xlims=(minSpeed, maxSpeed), ylims=(minProj, maxProj))
    #plt.title(cond1)
    #secondJoint.execute(xlims=(minSpeed, maxSpeed), ylims=(minProj, maxProj))
    #plt.title(cond2)

    firstExp = pd.DataFrame({'cond': cond1, 'Speed': firstJoint._speeds, 'Projection': firstJoint._projs})
    secExp = pd.DataFrame({'cond': cond2, 'Speed': secondJoint._speeds, 'Projection': secondJoint._projs})
    df = pd.concat((firstExp, secExp), ignore_index=True)

    cp = reversed(sns.dark_palette("purple", 2))
    sns.scatterplot(x='Speed', y='Projection', hue='cond', data=df, alpha=0.75, palette=cp)
    plt.xlabel('Speed [au / sec]')
    plt.ylabel('Projection [au / sec]')

    h1 = sns.jointplot(x='Speed', y='Projection', data=df[df['cond'] == cond1], kind='kde')
    #plt.xlabel('Speed [au / sec]')
    #plt.ylabel('Projection [au / sec]')

    h2 = sns.jointplot(x='Speed', y='Projection', data=df[df['cond'] == cond2], kind='kde')
    h1.set_axis_labels('Speed [au / sec]', 'Projection')
    h2.set_axis_labels('Speed [au / sec]', 'Projection')

    #plt.xlabel('Speed [au / sec]')
    #plt.ylabel('Projection [au / sec]')


    if showShow:
        plt.show()

    return df



def PairWiseRoi(cond1, firstExp, cond2, secondExp, showShow=True, show_count=True):
    #sns.set()
    plt.style.use("dark_background")
    sns.set_context("talk")

    print('Start Analyses..')
    firstRoi = RoiAnalysis(firstExp)
    secondRoi = RoiAnalysis(secondExp)

    firstRoi.execute()
    print('Finished First exp.')
    secondRoi.execute()
    print('Finished Second exp.')

    print(firstRoi._results)
    print(secondRoi._results)

    if show_count:
        plt.gca().plot(firstRoi._results['arrivedFrac'], label=" %s, %d worms" % (cond1, firstRoi._results['wormCount']))
        plt.gca().plot(secondRoi._results['arrivedFrac'], label=" %s, %d worms" % (cond2, secondRoi._results['wormCount']))
    else:
        plt.gca().plot(firstRoi._results['arrivedFrac'], label=cond1)
        plt.gca().plot(secondRoi._results['arrivedFrac'], label=cond2)

    plt.xlabel('Frames (2Hz)')
    plt.ylabel('Worms Arrived')
    plt.gca().legend(loc="lower right")
    plt.gca().grid(alpha=0.2)

    if showShow:
        plt.show()

def PairWiseProjectionDensity(cond1, firstExp, cond2, secondExp, showShow=True):
    plt.style.use("dark_background")
    sns.set_context("talk")

    LENGTH_THR = 50

    print('Start Analyses...')

    firstProj = [track.getMeanProjection(firstExp._regionsOfInterest['endReg']['pos']) for
                 track in firstExp._tracks if track._trackCords.shape[0] >= LENGTH_THR and track.getMaxDistTravelled() > 90]
    secondProj = [track.getMeanProjection(secondExp._regionsOfInterest['endReg']['pos']) for
                  track in secondExp._tracks if track._trackCords.shape[0] >= LENGTH_THR and track.getMaxDistTravelled() > 90]

    firstDf = pd.DataFrame({'proj' : firstProj, 'cond' : cond1})
    secondDf = pd.DataFrame({'proj': secondProj, 'cond': cond2})


    allProj = list(firstDf['proj'].values) + list(secondDf['proj'].values)
    allConds = list(firstDf['cond'].values) + list(secondDf['cond'].values)
    df = pd.DataFrame({'proj' : allProj, 'cond' : allConds})


    sns.kdeplot(firstDf['proj'], shade=True, label=cond1)
    ax = sns.kdeplot(secondDf['proj'], shade=True, label=cond2)

    plt.gca().grid(alpha=0.2)
    ax.set(xlabel="Projection", ylabel="Density")


    if showShow:
        plt.show()

    return df

def PairWiseSpeedDensity(cond1, firstExp, cond2, secondExp, showShow=True):
    LENGTH_THR = 250
    plt.style.use("dark_background")

    #sns.set()

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

    ax.set(xlabel="Speed [au / sec]", ylabel="Density")
    plt.gca().grid(alpha=0.2)

    if showShow:
        plt.show()
    #print(g)
    #g.draw()

    return df

def PairWiseOccupVisoulatizer(cond1, firstExp, cond2, secondExp):
    firstOccup = OccupVisualizer(firstExp)
    secondOccup = OccupVisualizer(secondExp)

    firstOccup.execute(cond1)
    secondOccup.execute(cond2)

def PairWiseRevDistances(cond1, firstExp, cond2, secondExp, showShow=True):
    rd1 = RevPosDensity(firstExp)
    rd1.execute(cond1)
    rd2 = RevPosDensity(secondExp)
    rd2.execute(cond2)

    if showShow:
        plt.show()

