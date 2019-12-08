# DEBUG
import sys
sys.path.append('/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/')
# DEBUG

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Behavior.Visualizers.RoiAnalysis import RoiAnalysis

from os import path

def meanProjectionMean(firstGroupExps, secondGroupExps, firstCondName, secondCondName):

    interestingTimePoints = [0,1000, 2000, 3000,4000]
    firstGroupResults = np.zeros((len(firstGroupExps), len(interestingTimePoints)))
    secoundGroupResults = np.zeros((len(secondGroupExps), len(interestingTimePoints)))

    for i, exp in enumerate(firstGroupExps):
        roiResults = RoiAnalysis(exp, trimTracksPos=(interestingTimePoints[-1] + 500))
        roiResults.execute()
        currentExpResults = roiResults._results['arrivedFrac']

        timePointResults = np.array(currentExpResults)[interestingTimePoints]
        firstGroupResults[i, :] = timePointResults

    for i, exp in enumerate(secondGroupExps):
        roiResults = RoiAnalysis(exp, trimTracksPos=(interestingTimePoints[-1] + 500))
        roiResults.execute()
        currentExpResults = roiResults._results['arrivedFrac']

        timePointResults = np.array(currentExpResults)[interestingTimePoints]
        secoundGroupResults[i, :] = timePointResults

    flattenFirstGroup = firstGroupResults.flatten(order='C')
    flattenSecondGroup = secoundGroupResults.flatten(order='C')

    fracitons = tuple(flattenFirstGroup) + tuple(flattenSecondGroup)
    conds = (firstCondName,) * flattenFirstGroup.size + (secondCondName,) * flattenSecondGroup.size

    timePoints = int((len(fracitons)) /
                     len(interestingTimePoints)) * tuple(interestingTimePoints)
    df = pd.DataFrame({'Fraction Arrived': fracitons,'Cond': conds, 'time': timePoints})

    sns.set(style='darkgrid')
    plt.style.use("dark_background")
    sns.pointplot(x='time', y='Fraction Arrived', data=df, hue='Cond')



    #DEBUG

def experiemntsFromFileList(fileList, condName):
    exps = [np.load(f)[0] for f in fileList]
    [exp.trimExperiment(4500) for exp in exps]

    print('Saving batch experiments..')
    #np.save(path.join('/home/itskov/Temp/', condName), exps,  allow_pickle=False)
    print('Done.')

    return exps

if __name__ == "__main__":
    firstGroupFiles = ['/mnt/storageNASRe/tph1/Results/28-Nov-2019/TPH_1_ATR_TRAIN_IAA3x5.avi_13.57.17/exp.npy',
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


    meanProjectionMean(experiemntsFromFileList(firstGroupFiles[0:1],'ATR_AND_IAA'),
                       experiemntsFromFileList(secondGroupFiles[0:1],'NO_ATR_AND_IAA'),
                       'ATR+(with IAA)', 'IAA-(with IAA)')
