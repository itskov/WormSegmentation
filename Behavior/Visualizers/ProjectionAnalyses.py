class ProjectionAnalyses:
    def __init__(self, exp):
        self._exp = exp
        self._results = {}

    def execute(self):
        LENGTH_THR = 250

        proj = [track.getMeanProjection(self._exp._regionsOfInterest['endReg']['pos']) / self._exp._scale for
               track in self._exp._tracks if
               track._trackCords.shape[0] >= LENGTH_THR and track.getMaxDistTravelled() > 350]

        self._results['proj'] = proj

