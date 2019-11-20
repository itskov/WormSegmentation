from Behavior.General.TracksFilter import filterTracksForAnalyses
import numpy as np


class RoiAnalysis:
    def __init__(self, exp):
        if (exp._scale == 1):
            exp.takeScale()

        if ('startReg' not in exp._regionsOfInterest):
            exp.addCirclePotisionRad('startReg', exp._scale / 5)
            exp.addCirclePotisionRad('endReg', exp._scale / 5)


        self._exp = exp
        self._results = {}


    def execute(self):
        print('Allocating space.')
        countStart = np.zeros((self._exp._numberOfFrames,))
        countEnd = np.zeros((self._exp._numberOfFrames,))

        for i, track in enumerate(self._exp._tracks):
            #print('Going over track %d' % i)
            frames = track._trackFrames
            distancesStart = track.getDistances(self._exp._regionsOfInterest['startReg']['pos'])
            distancesEnd = track.getDistances(self._exp._regionsOfInterest['endReg']['pos'])

            inRegion = distancesStart <=  self._exp._regionsOfInterest['startReg']['rad']
            outRegion = distancesStart > self._exp._regionsOfInterest['startReg']['rad']

            outEvents = inRegion[0:-2] & outRegion[1:-1]
            inEvents = inRegion[1:-1] & outRegion[0:-2]

            countStart[track._trackFrames[np.where(outEvents)]] -= 1
            countStart[track._trackFrames[np.where(inEvents)]] += 1

            # Doing the same for the end region
            inRegion = distancesEnd <=  self._exp._regionsOfInterest['endReg']['rad']
            outRegion = distancesEnd > self._exp._regionsOfInterest['endReg']['rad']

            outEvents = inRegion[0:-2] & outRegion[1:-1]
            inEvents = inRegion[1:-1] & outRegion[0:-2]

            countEnd[track._trackFrames[np.where(outEvents)]] -= 1
            countEnd[track._trackFrames[np.where(inEvents)]] += 1


            pass


        # Saving the results.
        self._results['wormCount'] = np.max(np.abs(np.cumsum(countStart)))
        self._results['arrived'] = np.cumsum(countEnd)
        self._results['arrivedFrac'] = self._results['arrived'] / float(self._results['wormCount'])


        #import matplotlib.pyplot as plt
        #plt.plot(self._results['arrivedFrac'])
        #pass


if __name__ == "__main__":
    import seaborn
    import matplotlib.pyplot as plt
    from Behavior.General import ExpDir

    # Setting the seaborn style.
    seaborn.set()

    #firstDir = '/home/itskov/Temp/05-Sep-2019/TPH_1_ATR_TRAIN_IAA3.avi_12.23.03'
    #secondDir = '/home/itskov/Temp/05-Sep-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_12.21.36'


    #firstDir = '/home/itskov/Temp/05-Sep-2019/TPH_1_ATR_TRAIN_IAA3.avi_14.47.06'
    #secondDir = '/home/itskov/Temp/05-Sep-2019/TPH_1_NO_ATR_TRAIN_IAA3.avi_14.47.22'


    firstDir = '/home/itskov/Temp/05-Sep-2019/TPH_1_ATR_TRAIN_NO_IAA3.avi_20.48.41'
    secondDir = '/home/itskov/Temp/05-Sep-2019/TPH_1_NO_ATR_TRAIN_NO_IAA3.avi_20.44.37'

    #firstDir = '/home/itskov/Temp/05-Sep-2019/TPH_1_ATR_TRAIN_NO_IAA3.avi_17.25.54'
    #secondDir = '/home/itskov/Temp/05-Sep-2019/TPH_1_NO_ATR_TRAIN_NO_IAA3.avi_17.24.56'

    firstAnalysis = RoiAnalysis(np.load(ExpDir(firstDir).getExpFile())[0])
    secondAnalysis = RoiAnalysis(np.load(ExpDir(secondDir).getExpFile())[0])

    firstAnalysis.execute()
    secondAnalysis.execute()

    plt.close('all')
    plt.plot(firstAnalysis._results['arrivedFrac']);
    plt.plot(secondAnalysis._results['arrivedFrac']);
    plt.xlabel('Frame (~ 1 [sec])')
    plt.ylabel('Worm Fraction')
    plt.legend(['ATR+ during training','ATR- during training'])
    plt.show()








