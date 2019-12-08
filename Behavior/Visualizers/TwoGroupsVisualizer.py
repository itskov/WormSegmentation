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

def meanProjectionMean(firstGroupName, secondGroupName, rootPath):

    firstExperimentsFiles = list(rootPath).rglob(Path(firstGroupName + '*/exp.npy'))
    secondExperimentsFiles = list(Path(rootPath).rglob(secondGroupName + '*/exp.npy'))

    interestingTimePoints = [0,1000, 2000, 3000,4000]
    firstGroupResults = np.zeros((len(firstExperimentsFiles), len(interestingTimePoints)))
    secoundGroupResults = np.zeros((len(secondExperimentsFiles), len(interestingTimePoints)))


    print('First Group:')
    for i, filename in enumerate(firstExperimentsFiles):
        print('Going over experiment: %s' % (filename,))

        # Getting the dirname of the experiment.
        expDirName = path.dirname(filename)
        arts = Artifacts(expDirName)

        roiResults = arts.getArtifact('roi')
        currentExpResults = roiResults._results['arrivedFrac']
        timePointResults = np.array(currentExpResults)[interestingTimePoints]
        firstGroupResults[i, :] = timePointResults

    print('Second Group:')
    for i, filename in enumerate(secondExperimentsFiles):
        print('Going over experiment: %s' % (filename,))

        # Getting the dirname of the experiment.
        expDirName = path.dirname(filename)
        arts = Artifacts(expDirName)

        roiResults = arts.getArtifact('roi')
        currentExpResults = roiResults._results['arrivedFrac']
        timePointResults = np.array(currentExpResults)[interestingTimePoints]
        secoundGroupResults[i, :] = timePointResults


    flattenFirstGroup = firstGroupResults.flatten(order='C')
    flattenSecondGroup = secoundGroupResults.flatten(order='C')

    fracitons = tuple(flattenFirstGroup) + tuple(flattenSecondGroup)
    conds = (firstGroupName,) * flattenFirstGroup.size + (secondGroupName,) * flattenSecondGroup.size

    timePoints = int((len(fracitons)) /
                     len(interestingTimePoints)) * tuple(interestingTimePoints)
    df = pd.DataFrame({'Fraction Arrived': fracitons,'Cond': conds, 'time': timePoints})

    sns.set(style='darkgrid')
    plt.style.use("dark_background")
    sns.pointplot(x='time', y='Fraction Arrived', data=df, hue='Cond')
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

    meanProjectionMean('TPH_1_ATR_TRAIN_IAA','TPH_1_NO_ATR_TRAIN_IAA','/mnt/storageNASRe/tph1/Results')
