from glob2 import glob
from os.path import join

class ExpDir:
    def __init__(self, expDir):
        self._expDir = expDir


    def getVidFile(self):
        files = glob(join(self._expDir,"*Compressed.mp4"))
        if len(files) == 1:
            return files[0]
        else:
            print('Error getting vid file: ' + str(files))


    def getTracksFile(self):
        files = glob(join(self._expDir, "*tracks.npy"))
        if len(files) == 1:
            return files[0]
        else:
            print('Error getting tracks file: ' + str(files))

    def getExpFile(self):
        files = glob(join(self._expDir, "exp.npy"))
        if len(files) == 1:
            return files[0]
        else:
            return None



