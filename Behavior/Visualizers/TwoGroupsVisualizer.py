import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Behavior.Visualizers.RoiAnalysis import RoiAnalysis

def meanProjectionMean(firstGroupExps, secondGroupExps, firstCondName, secondCondName):

    interestingTimePoints = [0,1000, 2000, 3000,4000,5000]
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

    #DEBUG
    secondGroupExps *= secondGroupExps * 1.2


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

if __name__ == "__main__":
    exp = np.load('/home/itskov/Temp/behav/28-Nov-2019/TPH_1_ATR[LIGHT]_TRAIN_FOOD_NO_IAA3x5.avi_19.49.59/exp.npy')[0]
    meanProjectionMean([exp,exp], [exp,exp])
