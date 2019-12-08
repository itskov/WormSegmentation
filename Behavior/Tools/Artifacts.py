import numpy as np

from os import path, mkdir

class Artifacts:
    def __init__(self, exp):
        _expDir = exp._expDir
        _currentDir = path.dirname(self._videoFilename)
        _artifactsDir = path.join(self._currentDir, 'artifacts')

    def checkForArtifactsDir(self):
        if not path.exists(self._artifactsDir):
            mkdir(self._artifactsDir)

    def addArtifact(self, name, obj):
        artifactName = path.join(self._artifactsDir, name)
        np.save(artifactName, [obj])




if __name__ == "__main__":
    import sys
    sys.path.append('/home/itskov/workspace/lab/DeepSemantic/WormSegmentation')

    from pathlib import Path
    from os import path
    from Behavior.Tools.Artifacts import Artifacts
    from Behavior.Visualizers.RoiAnalysis import RoiAnalysis
    from Behavior.Visualizers.ProjectionAnalyses import ProjectionAnalyses
    from Behavior.Visualizers.OccupVisualizer import OccupVisualizer

    import numpy as np

    rootDir = '/mnt/storageNASRe/tph1/Results/19-Nov-2019/'

    for fileName in Path(rootDir).rglob('exp.npy'):
        print(fileName)

        # load experiment.
        exp = np.load(fileName)[0]

        # Create an artifact folder.
        art = Artifacts(exp)

        # Checking for artifact dirs.
        art.checkForArtifactsDir()

        roiAnalyses = RoiAnalysis(exp)
        projectionAnalyses = ProjectionAnalyses(exp)
        occupAnalyses = OccupVisualizer(exp)

        art.addArtifact('roi', roiAnalyses._results)
        art.addArtifact('proj', projectionAnalyses._results)
        art.addArtifact('occup', occupAnalyses._results)
