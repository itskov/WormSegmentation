from ftplib import FTP
from os import path, mkdir

import re
import os
import sys
import time


def main():
    userName = 'itskov'
    pwd = 'Password1'
    extension = 'mj2'

    remotePath = sys.argv[1]
    localPath = sys.argv[2]

    # Connecting to data host
    print('Connecting..')
    ftp = FTP('132.64.59.3')
    ftp.login(userName, pwd)
    print('Connected.')
    ftp.cwd(remotePath)

    remoteFiles = ftp.nlst()
    remoteFiles = [file for file in remoteFiles if file.find('.' + extension) != -1]

    print('Going over the following files: %s' % str(remoteFiles))

    # Iterate over all of the files.
    for fileName in remoteFiles:
        print('Current file: %s' % fileName)
        # First extract features from filename
        fileDate = re.findall('\d\d-\w\w\w-\d\d\d\d', fileName)
        expTime = re.findall('-(\d\d\.\d\d\.\d\d)-', fileName)
        expName = re.findall('-M\w\w\d-(.*)\.', fileName)


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
        if not path.isdir(outputLocalDir):
            mkdir(outputLocalDir)

        # Creating the final destination of the file.
        outputFile = path.join(outputLocalDir, fileName)

        if os.path.exists(outputFile):
            print('File Exists. Continuing.')
        else:
            with open(outputFile,'wb') as fileHandle:
                print('Start retrieving.')
                retTime = time.time()
                ftp.retrbinary('RETR %s' % fileName, fileHandle.write)
                retTime = time.time() - retTime
                print('Done retrieving. in %d minutes' % (retTime / 60))

        #conduct(outputLocalDir)
        os.system('sbatch --mem=64g --gres gpu:m60:1 -c2 --time=0-12 ./processVideo.bash %s' % outputLocalDir)













if __name__ ==  "__main__":
    main()







