from glob import glob, escape
from os.path import join, isfile, exists

class ExpDir:
    def __init__(self, expDir):
        self._expDir = expDir


    def getVidFile(self):
        files = glob(join(escape(self._expDir),"*Compressed.mp4"))
        if len(files) == 1:
            return files[0]
        else:
            files = glob(join(escape(self._expDir), "*Full.mp4"))
            if len(files) == 1:
                return files[0]
            else:
                print('Error getting vid file: ' + str(files))


    def getTracksFile(self):
        files = glob(join(escape(self._expDir), "*tracks.npy"))
        if len(files) == 1:
            return files[0]
        else:
            print('Error getting tracks file: ' + str(files))

    def getExpFile(self):
        files = glob(join(escape(self._expDir), "exp.npy"))
        if len(files) == 1:
            return files[0]
        else:
            return None

    def getExpSegVid(self):
        files = glob(join(escape(self._expDir), "*seg.mp4"))
        if len(files) == 1:
            return files[0]
        else:
            return None


    def isValid(self):
        if (exists(self.getVidFile()) and exists(self.getTracksFile()) and exists(self.getExpFile())):
            return True

        return False



