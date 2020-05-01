class ProjectionAnalyses:
    def __init__(self, exp):
        self._exp = exp
        self._results = {}

    def execute(self):
        LENGTH_THR = 50

        proj = [track.getMeanProjection(self._exp._regionsOfInterest['endReg']['pos']) for
               track in self._exp._tracks if
               track._trackCords.shape[0] >= LENGTH_THR and track.getMaxDistTravelled() > 75]

        self._results['proj'] = proj

