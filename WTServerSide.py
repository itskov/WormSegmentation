import uuid
import re
import os

import numpy as np

# Flask
from flask import Flask, render_template, url_for
from glob2 import glob


from os.path import join
from os import mkdir, rmdir
from shutil import copyfile, copytree, rmtree
from scipy.ndimage import imread, label
from PIL import Image


class WTServerSide:
    def __init__(self):
        self.TEMP_DIRECTORY = './static/TempData'
        self.RAW_DATA = './static/RawData'
        self.SAVE_DIRECTORY = './static/SavedData'

    def createSessionDir(self):
        sessionId = str(uuid.uuid4())
        newTempPath = join(self.TEMP_DIRECTORY, sessionId)
        mkdir(newTempPath)

        # All the available raw samples.
        availRawSamples = self.getAvialableRawDataSamples()

        # Sampling a raw data point.
        rawSampleId = np.random.choice(availRawSamples, 1)[0]

        # Copying one raw sample into the temp directory.
        copyfile(join(self.RAW_DATA, str(rawSampleId) + ".orig.png"), join(newTempPath, str(rawSampleId) + ".orig.png"))
        copyfile(join(self.RAW_DATA, str(rawSampleId) + ".bw.png"), join(newTempPath, str(rawSampleId) + ".bw.png"))
        #copyfile(join(self.RAW_DATA, str(rawSampleId) + ".bw.png"), join(newTempPath, str(rawSampleId) + ".filter.png"))

        filteredImage = Image.fromarray(np.zeros((100,100), dtype=np.uint8))
        filteredImage.save(join(newTempPath, str(rawSampleId) + ".filter.png"), compress_level=0)

        # Returning the valid session id.
        return sessionId, rawSampleId

    def getSessionImages(self, sessionId):
        sessionPath = join(self.TEMP_DIRECTORY, sessionId, '*.png')

        # Removing the '.' in front of each filename.
        sessionFiles = sorted([f.replace('./static/','') for f in glob(sessionPath)])
        urls = [url_for('static', filename=f) for f in sessionFiles]


        return urls

    def saveSession(self, sessionId):
        sessionPath = join(self.TEMP_DIRECTORY, sessionId)
        rmtree(join(self.SAVE_DIRECTORY, sessionId), ignore_errors=True)
        copytree(sessionPath, join(self.SAVE_DIRECTORY, sessionId))



    def getAvialableRawDataSamples(self):
        # Copy raw data to session directory.
        # First we enumerate the files on the raw data directory.
        rawFiles = glob(join(self.RAW_DATA, '*.png'))

        if np.mod(len(rawFiles), 2) != 0:
            print('Error: Wrong number of raw samples.')
            return None

        # Orig raw files
        origRawFiles = glob(join(self.RAW_DATA, '*.orig.png'))

        # Get the ids of available samples.
        availRawSamples = np.unique([re.search('(\d+)\.orig.*', fn).group(1) for fn in origRawFiles])

        print(availRawSamples)

        # Returning the avilable raw samples.
        return [int(dig) for dig in availRawSamples]



    def updateSessionDir(self, sessionId, cordsY, cordsX):
        newTempPath = join(self.TEMP_DIRECTORY, sessionId)

        # Loading the images.
        sessionFiles = sorted(glob(join(newTempPath, '*.png')))
        print(sessionFiles)

        # Reading the bw image.
        bwImage = imread(sessionFiles[0])
        filteredImage = imread(sessionFiles[1])

        cordsY = int(cordsY * bwImage.shape[0])
        cordsX = int(cordsX * bwImage.shape[1])

        print('Cords: ' + str((cordsY, cordsX)))

        if (np.all(bwImage == filteredImage)):
            filteredImage = np.zeros(filteredImage.shape, dtype=np.uint8)

        labeledImage, n = label(bwImage)
        labeledImage = np.uint8(labeledImage);

        for i in range(1, n):
            blobsPosY, blobsPosX = np.where(labeledImage == i)
            if (cordsX in blobsPosX) and (cordsY in blobsPosY):
                if (bwImage[cordsY, cordsX] > 0):
                    print('i: ' + str(i) + ' Value: ' + str(bwImage[cordsY, cordsX]))
                    filteredImage[blobsPosY, blobsPosX] = bwImage[blobsPosY, blobsPosX]



            print('*** Labelling: ' + str(n) + " ****")
            print(np.unique(filteredImage))


        # Saving the new filtered file.
        newName = re.sub('\.filter\.png', '.filter_0.png', sessionFiles[1])
        newName = re.sub('_\d+', '_' + str(np.random.randint(0,100000)), newName)
        filteredImage = Image.fromarray(filteredImage)

        print('Writing ' + sessionFiles[1])
        filteredImage.save(newName, compress_leval=0)

        # Remove the old version of the filtered image.
        os.remove(sessionFiles[1])

    def removeSessionDir(self, sessionId):
        newTempPath = join(self.TEMP_DIRECTORY, sessionId)
        rmdir(newTempPath)

        pass


print ('Intializing..')
wtServerSide = WTServerSide()
staticPath = wtServerSide.TEMP_DIRECTORY[1:]
print('Static Path:' +  staticPath)

app = Flask(__name__)


@app.route('/<sessionId>')
@app.route('/')
def index(sessionId=None):


    if sessionId is None:
        sessionId, rawSampleId = wtServerSide.createSessionDir()
        print('Created new session: ' + sessionId + ' SampleId:' + str(rawSampleId))

    # Getting the images for this session.
    [bwImage, filteredImage, origImage] = wtServerSide.getSessionImages(sessionId)

    return render_template('MainTemplate.html', sessionId=sessionId,
                           origImage=origImage,
                           bwImage=bwImage,
                           filteredImage=filteredImage)


@app.route('/<sessionId>/<posY>/<posX>/<height>/<width>')
def updateSession(sessionId, posY, posX, height, width):
    wtServerSide.updateSessionDir(sessionId, float(posY) / float(height), float(posX) / float(width))

    return index(sessionId)

@app.route('/<sessionId>/save')
def saveSession(sessionId):
    wtServerSide.saveSession(sessionId)

    return index(sessionId)

#app.secret_key = 'any random string’

if __name__ == "__main__":
    index('965d3e6d-d4a8-4a5c-9ed6-e3ae6d1278ef')








