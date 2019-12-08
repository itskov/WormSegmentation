if __name__ == "__main__":
    import sys
    sys.path.append('c:\\Eyal\\WormSegmentation')

    from pathlib import Path
    from os import path
    from Behavior.Tools.Artifacts import Artifacts
    from Behavior.Visualizers.RoiAnalysis import RoiAnalysis
    from Behavior.Visualizers.ProjectionAnalyses import ProjectionAnalyses
    from Behavior.Visualizers.OccupVisualizer import OccupVisualizer

    from Behavior.General.ExpDir import ExpDir

    import numpy as np

    rootDir = '\\\\132.64.59.87\\home\\tph1\\Results\\'

    for fileName in Path(rootDir).rglob('exp.npy'):
        try:
            print(fileName)
            dirName = path.dirname(fileName)

            # load experiment.
            exp = np.load(fileName, allow_pickle=True)[0]
            print(exp._scale)
            print(exp._regionsOfInterest)

            if exp._scale == 1:
                  print('Bad scale. skipping')
                  continue

            exp.initialize(ExpDir(dirName))	
            exp.trimExperiment(4500)

            # Create an artifact folder.
            art = Artifacts(exp, dirName)

            # Checking for artifact dirs.
            art.checkForArtifactsDir()

            roiAnalyses = RoiAnalysis(exp)
            projectionAnalyses = ProjectionAnalyses(exp)
            occupAnalyses = OccupVisualizer(exp)

            roiAnalyses.execute()
            projectionAnalyses.execute()
            occupAnalyses.execute(showPlot=False)

            print('Saving the Artifcats.')
            art.addArtifact('roi', roiAnalyses._results)
            art.addArtifact('proj', projectionAnalyses._results)
            art.addArtifact('occup', occupAnalyses._results)
        except:
            print('Error with %s' % (fileName,))
