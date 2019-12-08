import numpy as np

from os import path, mkdir

class Artifacts:
    def __init__(self, exp=None, expLocation=None):
        if exp is not None:
            self._currentDir = path.dirname(exp._videoFilename)
        elif expLocation is not None:
            self._currentDir = expLocation
        else:
            print('Error: Cant instaitiate Artifacts.')

        self._artifactsDir = path.join(self._currentDir, 'artifacts')

    def checkForArtifactsDir(self):
        if not path.exists(self._artifactsDir):
            mkdir(self._artifactsDir)

    def addArtifact(self, name, obj):
        artifactName = path.join(self._artifactsDir, name)
        np.save(artifactName, [obj])

    def getArtifact(self, name):
        artifactName = path.join(self._artifactsDir, name)
        return np.load(artifactName)[0]





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

    rootDir = '/mnt/storageNASRe/tph1/Results/28-Nov-2019/'

    for fileName in Path(rootDir).rglob('exp.npy'):
        print(fileName)

        # load experiment.
        exp = np.load(fileName)[0]
        exp.trimExperiment(4500)

        # Create an artifact folder.
        art = Artifacts(exp)

        # Checking for artifact dirs.
        art.checkForArtifactsDir()

        roiAnalyses = RoiAnalysis(exp)
        projectionAnalyses = ProjectionAnalyses(exp)
        occupAnalyses = OccupVisualizer(exp)

        roiAnalyses.execute()
        projectionAnalyses.execute()
        occupAnalyses.execute(showPlot=False)


        art.addArtifact('roi', roiAnalyses._results)
        art.addArtifact('proj', projectionAnalyses._results)
        art.addArtifact('occup', occupAnalyses._results)
