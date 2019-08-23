from ftplib import FTP
from os import path, mkdir

import re
import sys

from conductor import conduct


def main():
    userName = 're'
    pwd = 're'
    extension = 'mat'

    remotePath = sys.argv[1]
    localPath = '/home/itskov/Temp'

    # Connecting to data host
    ftp = FTP('132.64.59.87')
    ftp.login(userName, pwd)
    ftp.cwd(remotePath)

    remoteFiles = ftp.nlst()
    remoteFiles = [file for file in remoteFiles if file.find('.' + extension) != -1]

    # Iterate over all of the files.
    for fileName in remoteFiles:
        # First extract features from filename
        fileDate = re.findall('\d\d-\w\w\w-\d\d\d\d', fileName)
        expTime = re.findall('-(\d\d\.\d\d\.\d\d)-', fileName)
        expName = re.findall('-Mic\d-(.*)\.', fileName)


        # Checking the format.
        if len(fileDate) == 0 or len(expTime) == 0 or len(expName) == 0:
            print('ERROR: Bad file format: %s' % fileName)
            return

        fileDate = fileDate[0]
        expTime = expTime[0]
        expName = expName[0]

        # Output parent dir
        outputParentDir = path.join(localPath,fileDate)
        if not path.isdir(outputParentDir):
            mkdir(outputParentDir)

        # Creating the localdir
        outputLocalDir = path.join(outputParentDir, expName + "_" + expTime)
        mkdir(outputLocalDir)

        # Creating the final destination of the file.
        outputFile = path.join(outputLocalDir, fileName)
        with open(outputFile,'wb') as fileHandle:
            ftp.retrbinary('RETR %s' % fileName, fileHandle.write)

        conduct(outputLocalDir)













if __name__ ==  "__main__":
    main()







