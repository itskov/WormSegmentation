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
        countStart = np.zeros((self._exp._numberOfFrames,))
        countEnd = np.zeros((self._exp._numberOfFrames,))

        for track in self._exp._tracks:
            frames = track._trackFrames
            distancesStart = track.getDistances(self._exp._regionsOfInterest['startReg']['pos'])
            distancesEnd = track.getDistances(self._exp._regionsOfInterest['endReg']['pos'])

            inRegion = distancesStart <=  self._exp._regionsOfInterest['startReg']['rad']
            outRegion = distancesStart > self._exp._regionsOfInterest['startReg']['rad']

            outEvents = inRegion[0:-2] & outRegion[1:-1]
            inEvents = inRegion[1:-1] & outRegion[0:-2]

            countStart[np.where(outEvents)] -= 1
            countStart[np.where(inEvents)] += 1

            # Doing the same for the end region
            inRegion = distancesEnd <=  self._exp._regionsOfInterest['endReg']['rad']
            outRegion = distancesEnd > self._exp._regionsOfInterest['endReg']['rad']

            outEvents = inRegion[0:-2] & outRegion[1:-1]
            inEvents = inRegion[1:-1] & outRegion[0:-2]

            countEnd[np.where(outEvents)] -= 1
            countEnd[np.where(inEvents)] += 1


            pass


        # Saving the results.
        self._results['wormCount'] = np.max(np.abs(np.cumsum(countStart)))
        self._results['arrived'] = np.cumsum(countEnd)
        self._results['arrivedFrac'] = self._results['arrived'] / float(self._results['wormCount'])


        import matplotlib.pyplot as plt
        plt.plot(self._results['arrivedFrac'])
        pass







