class SpeedsAnalyses:
    def __init__(self, exp):
        self._exp = exp
        self._results = {}

    def execute(self):
        LENGTH_THR = 60

        speeds = [track.getMeanSpeed() / self._exp._scale for
                     track in self._exp._tracks if track._trackCords.shape[0] >= LENGTH_THR]

        self._results['speed'] = speeds


