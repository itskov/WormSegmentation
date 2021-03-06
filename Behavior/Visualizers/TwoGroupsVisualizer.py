# DEBUG
import sys
sys.path.append('/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/')
# DEBUG

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Behavior.Visualizers.RoiAnalysis import RoiAnalysis
from Behavior.Tools.Artifacts import Artifacts

from pathlib import Path
from os import path

# Plotting the with IAA plots
def meanProjectionMeanWithIaa(firstGroupName, secondGroupName, rootPath):
    firstExperimentsFiles = list(Path(rootPath).rglob(firstGroupName + '*/exp.npy'))
    secondExperimentsFiles = list(Path(rootPath).rglob(secondGroupName + '*/exp.npy'))

    interestingTimePoints = [0,1000, 2000, 3000,4000]
    firstGroupResults = np.zeros((len(firstExperimentsFiles), len(interestingTimePoints)))
    secoundGroupResults = np.zeros((len(secondExperimentsFiles), len(interestingTimePoints)))


    print('First Group:')
    for i, filename in enumerate(firstExperimentsFiles):
        if 'Nov' not in str(filename):
            continue

        if '03-Nov' in str(filename):
            continue

        # Getting the dirname of the experiment.
        expDirName = path.dirname(filename)
        arts = Artifacts(expLocation=expDirName)

        roiResults = arts.getArtifact('roi')

        if roiResults == None:
            continue

        print('Going over experiment: %s' % (filename,))

        currentExpResults = roiResults['arrivedFrac']
        timePointResults = np.array(currentExpResults)[interestingTimePoints]
        firstGroupResults[i, :] = timePointResults

    print('Second Group:')
    for i, filename in enumerate(secondExperimentsFiles):
        if 'Nov' not in str(filename):
            continue

        if '03-Nov' in str(filename):
            continue

        # Getting the dirname of the experiment.
        expDirName = path.dirname(filename)
        arts = Artifacts(expLocation=expDirName)

        roiResults = arts.getArtifact('roi')

        if roiResults == None:
            continue

        print('Going over experiment: %s' % (filename,))

        currentExpResults = roiResults['arrivedFrac']
        timePointResults = np.array(currentExpResults)[interestingTimePoints]
        secoundGroupResults[i, :] = timePointResults

    firstGroupResults = firstGroupResults[np.logical_not(np.all(firstGroupResults == 0, axis=1)),:]
    secoundGroupResults = secoundGroupResults[np.logical_not(np.all(secoundGroupResults == 0, axis=1)),:]

    flattenFirstGroup = firstGroupResults.flatten(order='C')
    flattenSecondGroup = secoundGroupResults.flatten(order='C')

    fracitons = tuple(flattenFirstGroup) + tuple(flattenSecondGroup)
    conds = (firstGroupName,) * flattenFirstGroup.size + (secondGroupName,) * flattenSecondGroup.size

    timePoints = int((len(fracitons)) /
                     len(interestingTimePoints)) * tuple(interestingTimePoints)
    df = pd.DataFrame({'Fraction Arrived': fracitons,'Cond': conds, 'time': timePoints})

    sns.set(style='darkgrid')
    plt.style.use("dark_background")
    ax = sns.pointplot(x='time', y='Fraction Arrived', data=df, hue='Cond')
    ax.set(xlabel='Frames [2Hz]')
    plt.show()

    pairedDf_2 = pd.DataFrame({'ATR+' : firstGroupResults[[1,2,3,4,5,6,7,9],2], 'ATR-': secoundGroupResults[[2,3,4,5,6,7,8,9],2]})
    pairedDf_3 = pd.DataFrame({'ATR+': firstGroupResults[[1, 2, 3, 4, 5, 6, 7, 9], 3], 'ATR-': secoundGroupResults[[2, 3, 4, 5, 6, 7, 8, 9], 3]})
    pairedDf_4 = pd.DataFrame({'ATR+': firstGroupResults[[1, 2, 3, 4, 5, 6, 7, 9], 4], 'ATR-': secoundGroupResults[[2, 3, 4, 5, 6, 7, 8, 9], 4]})
    pairedDf = pd.concat((pairedDf_2,pairedDf_3,pairedDf_4), ignore_index=True)

    ax = sns.scatterplot(x='ATR-', y='ATR+',data=pairedDf)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, ':k', color='white')

    print('Done.')

# Plotting the with IAA plots
def meanProjectionMeanWithoutIaa(firstGroupName, secondGroupName, rootPath):
    firstExperimentsFiles = list(Path(rootPath).rglob(firstGroupName + '*/exp.npy'))
    secondExperimentsFiles = list(Path(rootPath).rglob(secondGroupName + '*/exp.npy'))

    interestingTimePoints = [0,1000, 2000, 3000,4000]
    firstGroupResults = np.zeros((len(firstExperimentsFiles), len(interestingTimePoints)))
    secoundGroupResults = np.zeros((len(secondExperimentsFiles), len(interestingTimePoints)))


    print('First Group:')
    for i, filename in enumerate(firstExperimentsFiles):
        if 'Nov' not in str(filename):
            continue

        # Getting the dirname of the experiment.
        expDirName = path.dirname(filename)
        arts = Artifacts(expLocation=expDirName)

        roiResults = arts.getArtifact('roi')

        if roiResults == None:
            continue

        print('Going over experiment: %s' % (filename,))

        currentExpResults = roiResults['arrivedFrac']
        timePointResults = np.array(currentExpResults)[interestingTimePoints]
        firstGroupResults[i, :] = timePointResults

    print('Second Group:')
    for i, filename in enumerate(secondExperimentsFiles):
        if 'Nov' not in str(filename):
            continue


        # Getting the dirname of the experiment.
        expDirName = path.dirname(filename)
        arts = Artifacts(expLocation=expDirName)

        roiResults = arts.getArtifact('roi')

        if roiResults == None:
            continue

        print('Going over experiment: %s' % (filename,))

        currentExpResults = roiResults['arrivedFrac']
        timePointResults = np.array(currentExpResults)[interestingTimePoints]
        secoundGroupResults[i, :] = timePointResults

    firstGroupResults = firstGroupResults[np.logical_not(np.all(firstGroupResults == 0, axis=1)),:]
    secoundGroupResults = secoundGroupResults[np.logical_not(np.all(secoundGroupResults == 0, axis=1)),:]

    flattenFirstGroup = firstGroupResults.flatten(order='C')
    flattenSecondGroup = secoundGroupResults.flatten(order='C')

    fracitons = tuple(flattenFirstGroup) + tuple(flattenSecondGroup)
    conds = (firstGroupName,) * flattenFirstGroup.size + (secondGroupName,) * flattenSecondGroup.size

    timePoints = int((len(fracitons)) /
                     len(interestingTimePoints)) * tuple(interestingTimePoints)
    df = pd.DataFrame({'Fraction Arrived': fracitons,'Cond': conds, 'time': timePoints})

    sns.set(style='darkgrid')
    plt.style.use("dark_background")
    ax = sns.pointplot(x='time', y='Fraction Arrived', data=df, hue='Cond')
    ax.set(xlabel='Frames [2Hz]')
    plt.show()

    pairedDf_2 = pd.DataFrame({'ATR+': firstGroupResults[[1, 2, 3, 4, 6], 2], 'ATR-': secoundGroupResults[[2, 4, 3, 5, 6], 2]})
    pairedDf_3 = pd.DataFrame({'ATR+': firstGroupResults[[1, 2, 3, 4, 6], 3], 'ATR-': secoundGroupResults[[2, 4, 3, 5, 6], 3]})
    pairedDf_4 = pd.DataFrame({'ATR+': firstGroupResults[[1, 2, 3, 4, 6], 4], 'ATR-': secoundGroupResults[[2, 4, 3, 5, 6], 4]})
    pairedDf = pd.concat((pairedDf_2,pairedDf_3,pairedDf_4), ignore_index=True)

    ax = sns.scatterplot(x='ATR-', y='ATR+',data=pairedDf)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.plot(lims, lims, ':k', color='white')
    plt.show()

    print('Done.')





if __name__ == "__main__":
    '''firstGroupFiles = ['/mnt/storageNASRe/tph1/Results/28-Nov-2019/TPH_1_ATR_TRAIN_IAA3x5.avi_13.57.17/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/24-Nov-2019/TPH_1_ATR_TRAIN_IAA3x5.avi_18.11.35/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/19-Nov-2019/TPH_1_ATR_TRAIN_IAA3x5.avi_10.36.51/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/19-Nov-2019/TPH_1_ATR_TRAIN_IAA3x5.avi_13.22.04/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/17-Nov-2019/TPH_1_ATR_TRAIN_IAA5x3.avi_12.30.50/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/12-Nov-2019/TPH_1_ATR_TRAIN_IAA3.avi_12.05.51/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/07-Nov-2019/TPH_1_ATR_TRAIN_IAA3.avi_10.41.50/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/07-Nov-2019/TPH_1_ATR_TRAIN_IAA3.avi_13.54.33/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/06-Nov-2019/TPH_1_ATR_TRAIN_IAA3.avi_16.52.02/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/03-Nov-2019/TPH_1_ATR_TRAIN_IAA3.avi_11.41.32/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/03-Nov-2019/TPH_1_ATR_TRAIN_IAA3.avi_15.00.31/exp.npy',
                       '/mnt/storageNASRe/tph1/Results/27-Oct-2019/TPH_1_ATR_TRAIN_IAA3.avi_11.50.40/exp.npy',]

    secondGroupFiles = ['/mnt/storageNASRe/tph1/Results/28-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3x5.avi_13.56.35/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/24-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3x5.avi_18.10.50/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/19-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3x5.avi_10.36.03/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/19-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3x5.avi_13.21.14/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/17-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA5x3.avi_12.29.55/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/12-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_12.05.02/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/07-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_10.41.03/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/07-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_13.53.47/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/06-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_16.51.08/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/03-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_11.40.45/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/03-Nov-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_14.59.57/exp.npy',
                        '/mnt/storageNASRe/tph1/Results/27-Oct-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_11.49.55/exp.npy',]

    '''

    #meanProjectionMeanWithIaa('TPH_1_ATR_TRAIN_IAA','TPH_1_NO_ATR_TRAIN_IAA','/mnt/storageNASRe/tph1/Results')
    #meanProjectionMeanWithoutIaa('TPH_1_ATR_TRAIN_NO_IAA', 'TPH_1_NO_ATR_TRAIN_NO_IAA', '/mnt/storageNASRe/tph1/Results')
    from Behavior.General.ExpPair import ExpPair
    from Behavior.Visualizers.PairwiseAnalyses import PairWiseProjectionDensity, PairWiseRoi

    expPair = ExpPair('//home/itskov/Temp/behav/04-Dec-2019/TPH_1_ATR_TRAIN_FOOD_AND_NO_IAA3x5.avi_17.15.11/exp.npy',
                      '/home/itskov/Temp/behav/04-Dec-2019/TPH_1_NO_ATR_TRAIN_FOOD_AND_NO_IAA3x5.avi_17.14.13/exp.npy')

    expPair._firstExp.trimExperiment(4501)
    expPair._secondExp.trimExperiment(4501)

    PairWiseRoi('ATR+ with Food', expPair._firstExp,'ATR- with Food', expPair._secondExp)
    PairWiseProjectionDensity('ATR+ with Food', expPair._firstExp,'ATR- with Food', expPair._secondExp)

    pass






