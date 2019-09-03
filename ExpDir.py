from glob2 import glob

class ExpDir:
    def __init__(self, expDir):
        self._expDir = expDir


    def getVidFile(self):
        files = glob(self._expDir + "*Full.mp4")
        if (len(files) == 1):
            return files[0]
        else:
            print('Error getting vid file: ' + files)


    def getTracksFile(self):
        files = glob(self._expDir + "*.npy")
        if (len(files) == 1):
            return files[0]
        else:
            print('Error getting tracks file: ' + files)


