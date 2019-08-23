import sys


from glob2 import glob
from os import path, remove
from ProcessVideo import Process
from SegmentedTracker import SegmentedTracker

from SplitChannels import splitChannel


def conduct(inputDir = None):
    RESTORE_POINT = "/cs/phd/itskov/WormSegmentation/WormSegmentatioNetworks/WormSegmentation"

    # getting the input directory
    inputDir = sys.argv[1] if inputDir == None else inputDir

    mj2Files = glob(path.join(inputDir, "*.mj2"))
    if (len(mj2Files) > 1):
        print('Error: ambiguous mj2 files.')
        return

    if (len(mj2Files) == 1):
        mj2File = path.join(inputDir, mj2Files[0])
        print('Working on a mj2 file: %s' % mj2File)
        splitChannel((mj2File, 0))
        print('Removing mj2.')
        remove(mj2File)

    files = glob(path.join(inputDir, "*_Full.mp4"))

    if (len(files) > 1):
        print('Error: Ambiguous input file.')
        #print(files)
        return

    inputFile = files[0]

    print('Initial file: ' + inputFile )

    outputFile = Process(RESTORE_POINT, inputFile)
    print('Tracking: ' + outputFile)
    segTracker = SegmentedTracker(outputFile, inputFile)
    segTracker.track()
    segTracker.filterTracks()
    segTracker.saveTracks()
    segTracker.createTrackedMovie()








if __name__ == "__main__":
    conduct()