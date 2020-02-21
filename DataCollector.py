import cv2

import numpy as np
import matplotlib.pyplot as plt

from glob2 import glob
from PIL import Image
from scipy import stats
from os.path import join


class DataCollector:
    def __init__(self):
        print('Starting..')
        self._GLOB_TERM = '/home/itskov/Temp/behav/20-Feb-2020/**/*Full.mp4'

        # Saving the paths video file.
        self._videoFiles = glob(self._GLOB_TERM)

        # The coordinate of the small images
        self._SNIP_SIZE = (100, 100)

        # If the user wants a smaller region
        self._regions = None

        print('Found %d relevant files.' % len(self._videoFiles))

    def setRegions(self):
        for i, video_file in enumerate(self._videoFiles):
            cap = cv2.VideoCapture(video_file)
            _, frame = cap.read()
            plt.imshow(frame)
            new_points = plt.ginput(2, timeout=-1)
            new_points = np.fliplr(new_points).astype(np.int)
            self._regions = [new_points] if self._regions is None else self._regions.append(new_points)
            plt.close()

            cap.release()





    def fetchImage(self):
        # Fetching a file to read from.
        choiced = np.random.choice(range(len(self._videoFiles)))

        fetchedFile = self._videoFiles[choiced]

        # Read file
        cap = cv2.VideoCapture(fetchedFile)

        # Get frames number
        movieLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # DEBUG
        movieLength *= 0.8
        # DEBUG
        #length = 5000

        # Sample frame
        #frameNum = np.random.choice(range(length))

        # Sample from a truncated normal distribution.
        frameNum = stats.truncnorm(-1,1).rvs()
        frameNum = np.round(frameNum * (movieLength / 2) + movieLength / 2)


        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        success, readFrame = cap.read()

        if self._regions is not None:
            region = self._regions[choiced]

            readFrame = readFrame[region[0,0]:region[1,0], region[0,1]:region[1,1]]


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

        #DEBUG
        #heightPos = stats.truncnorm(0.2, 0.7).rvs()
        #widthPos = stats.truncnorm(0.2, 0.7).rvs()


        heightPos = int(np.round(heightPos * (height/2 - self._SNIP_SIZE[0]) + height/2))
        widthPos = int(np.round(widthPos * (width / 2 - self._SNIP_SIZE[1]) + width / 2))



        #plt.imshow(curImage)
        #d = plt.ginput(1)


        smallImage = curImage[heightPos:(heightPos + self._SNIP_SIZE[0]),
                     widthPos:(widthPos + self._SNIP_SIZE[1])]

        smallImage = cv2.cvtColor(smallImage, cv2.COLOR_BGR2GRAY)

        return smallImage

    def getSeg(self, curImage):
        if curImage is None:
            return None

        thImage = cv2.adaptiveThreshold(curImage, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY, 17, 2)

        thImage = np.abs(thImage - 1)

        return thImage



from multiprocessing import Pool

dc = DataCollector()
dc.setRegions()

def saveImage(i):

    try:
        im = dc.getSnip()
        imt = dc.getSeg(im)

        if imt is not None:
            print('Got snip!')

            im_I = Image.fromarray(im)
            imt_I = Image.fromarray(imt)

            im_I.save(join('./static/RawData/', str(i) + '.orig.png'), compress_level=0)
            imt_I.save(join('./static/RawData/', str(i) + '.bw.png'), compress_level=0)
    except Exception:
        print('Error creating sample')
        raise;



    print(str(i))


if __name__ == "__main__":
    print('Running threads..')
    with Pool(processes=4) as pool:
        pool.map(saveImage, np.random.choice(range(1,5*10**6),100000))
    print('Done.')
    #saveImage(4)















