from Behavior.General.TracksFilter import filterTracksForAnalyses
import numpy as np

class RunsLengthAnalyses:
    def __init__(self, exp):
        self._exp = exp
        self._results = {}

    def execute(self):
        THR = 5
        tracks = filterTracksForAnalyses(self._exp._tracks, minDistance=150, minSteps=100)

        runs_length = np.concatenate([np.ravel(t.getRunsLength()) for t in tracks])
        runs_length = runs_length[runs_length > THR]

        self._results['run_lens'] = runs_length


