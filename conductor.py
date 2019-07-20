import sys


from glob2 import glob
from os import path
from ProcessVideo import Process
from SegmentedTracker import SegmentedTracker



def main():
    RESTORE_POINT = "/cs/phd/itskov/WormSegmentation/WormSegmentatioNetworks/WormSegmentation"

    # getting the input directory
    inputDir = sys.argv[1]
    files = glob(path.join(inputDir, "*_Full.mp4"))

    if (len(files) > 1):
        print('Error: Ambiguous input file.')
        print(files)
        return

    inputFile = files[0]

    print('Initial file: ' + inputFile )

    outputFile = Process(RESTORE_POINT, inputFile)
    print('Tracking: ' + outputFile)
    segTracker = SegmentedTracker(outputFile, inputFile)
    segTracker.track()
    segTracker.saveTracks()
    segTracker.createTrackedMovie()








if __name__ == "__main__":
    main()