import cv2

import numpy as np
import matplotlib.pyplot as plt

from glob2 import glob
from PIL import Image
from scipy import stats
from os.path import join


class DataCollector:
    def __init__(self):
        self._GLOB_TERM = '/mnt/storageNASRe/tph1/26*Full.mp4'

        # Saving the paths video file.
        self._videoFiles = glob(self._GLOB_TERM)

        # The coordinate of the small images
        self._SNIP_SIZE = (100, 100)


    def fetchImage(self):
        # Fetching a file to read from.
        fetchedFile = np.random.choice(self._videoFiles)

        # Read file
        cap = cv2.VideoCapture(fetchedFile)

        # Get frames number
        movieLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # DEBUG
        movieLength *= 0.6
        # DEBUG
        #length = 5000

        # Sample frame
        #frameNum = np.random.choice(range(length))

        # Sample from a truncated normal distribution.
        frameNum = stats.truncnorm(-1,1).rvs()
        frameNum = np.round(frameNum * (movieLength / 2) + movieLength / 2)


        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        success, readFrame = cap.read()

        return readFrame, fetchedFile, frameNum


    def getSnip(self):
        curImage, fileName, frameNum = self.fetchImage()
        print('Fetched frame: ' + str(frameNum) + ' from ' + fileName)

        if (curImage is None):
            return None

        # Getting image shape
        height, width, channels = curImage.shape


        #heightPos = np.random.randint(height - (self._SNIP_SIZE[0]))
        #widthPos = np.random.randint(width - (self._SNIP_SIZE[1]))
        # Sampling from truncated normal variable.
        heightPos = stats.truncnorm(-1, 1).rvs()
        widthPos = stats.truncnorm(-1, 1).rvs()
        heightPos = int(np.round(heightPos * (height/2 - self._SNIP_SIZE[0]) + height/2))
        widthPos = int(np.round(widthPos * (width / 2 - self._SNIP_SIZE[1]) + width / 2))

        smallImage = curImage[heightPos:(heightPos + self._SNIP_SIZE[0]),
                     widthPos:(widthPos + self._SNIP_SIZE[1])]

        smallImage = cv2.cvtColor(smallImage, cv2.COLOR_BGR2GRAY)

        return smallImage

    def getSeg(self, curImage):
        thImage = cv2.adaptiveThreshold(curImage, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY, 11, 2)

        thImage = np.abs(thImage - 1)

        return thImage



from multiprocessing import Pool

dc = DataCollector()

def saveImage(i):

    try:
        im = dc.getSnip()
        imt = dc.getSeg(im)

        print('Got snip!')

        im_I = Image.fromarray(im)
        imt_I = Image.fromarray(imt)

        im_I.save(join('./static/RawData/', str(i) + '.orig.png'), compress_level=0)
        imt_I.save(join('./static/RawData/', str(i) + '.bw.png'), compress_level=0)
    except Exception:
        print('Error creating sample')


    print(str(i))


if __name__ == "__main__":
    with Pool(processes=4) as pool:
        pool.map(saveImage, np.random.choice(range(1,5*10**6),100000))
    #saveImage(4)















