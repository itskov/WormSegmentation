import sys

from glob import glob, escape
from os import path, remove
from ProcessVideo import Process
from SegmentedTracker import SegmentedTracker

from SplitChannels import splitChannel


def conduct(inputDir = None):
    RESTORE_POINT = "/cs/phd/itskov/WormSegmentation/WormSegmentatioNetworks/WormSegmentation"

    # getting the input directory
    inputDir = sys.argv[1] if inputDir is None else inputDir

    # Lookign to see if the experiment is not ready already.
    tracksFiles = glob(path.join(escape(inputDir), "*tracks.npy"))

    if len(tracksFiles) > 0:
        print('--== Tracks already analyzed. Exiting. ==--')
        return


    mj2Files = glob(path.join(escape(inputDir), "*.mj2"))
    if (len(mj2Files) > 1):
        print('Error: ambiguous mj2 files.')
        return

    if (len(mj2Files) == 1):
        mj2File = path.join(inputDir, mj2Files[0])
        print('Working on a mj2 file: %s' % mj2File)
        splitChannel((mj2File, 0))
        print('Removing mj2.')
        remove(mj2File)

    files = glob(path.join(escape(inputDir), "*_Full.mp4"))

    if (len(files) > 1):
        print('Error: Ambiguous input file.')
        #print(files)
        return

    inputFile = files[0]

    print('Initial file: ' + inputFile)

    outputFile = Process(RESTORE_POINT, inputFile)
    print('Tracking: ' + outputFile)
    segTracker = SegmentedTracker(outputFile, inputFile)
    segTracker.track()
    segTracker.filterTracks()
    segTracker.saveTracks()
    segTracker.createTrackedMovie()








if __name__ == "__main__":
    conduct()
