import uuid

import numpy as np

from shutil import rmtree
from os import remove, mkdir
from os.path import join, isdir
from imageio import imread, imwrite
from PIL import Image

from glob2 import glob


class DataAugumenter:
    def __init__(self, fromDirectory, toDirectory):
        self._fromDirectory = fromDirectory
        self._toDirectory = toDirectory

        if isdir(self._toDirectory):
            rmtree(self._toDirectory)

        mkdir(self._toDirectory)

    def createTrainData(self):
        # Collecting the input data.
        entireSavedData = glob(join(self._fromDirectory, '*'))

        for i in range(len(entireSavedData)):
            currentDirectory = entireSavedData[i]

            sampleFiles = sorted(glob(join(currentDirectory, '*')))

            # Taking the correct filenames.
            filteredFileName = sampleFiles[1]
            origFileName = sampleFiles[2]

            # Reading the sample.
            filteredImage = Image.open(sampleFiles[1])
            origImage = Image.open(sampleFiles[2])

            tempFilt = filteredImage
            tempOrig = origImage

            for i in range(4):
                tempFilt = tempFilt.rotate(90)
                tempOrig = tempOrig.rotate(90)

                trainSampleName = str(uuid.uuid4())
                np.save(join(self._toDirectory, trainSampleName), [np.asanyarray(tempOrig), np.asanyarray(tempFilt)])
                pass




if __name__ == "__main__":
    da = DataAugumenter()
    da.createTrainData()
