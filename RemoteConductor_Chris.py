from ftplib import FTP
from os import path, mkdir


import re
import os
import sys
import time

from conductor import conduct


def main():
    userName = 'itskov'
    pwd = 'Password1'
    extension = 'mj2'

    remotePath = sys.argv[1]
    localPath = sys.argv[2]


    WAIT_HOURS = 0.75
    WAIT_INTERVAL = 6

    # Connecting to data host
    print('Connecting..')
    ftp = FTP('132.64.59.3')
    ftp.login(userName, pwd)
    print('Connected.')
    ftp.cwd(remotePath)

    remoteFiles = ftp.nlst()
    remoteFiles = [file for file in remoteFiles if file.find('.' + extension) != -1]

    print('Going over the following files: %s' % str(remoteFiles))


    workedFiles = 0

    # Iterate over all of the files.
    for i, fileName in enumerate(remoteFiles):
        timeOfDay = "%2d:%2d" % (time.localtime().tm_hour, time.localtime().tm_min)
        print(('Current file: %s' % fileName) + " time: " + timeOfDay)
        # First extract features from filename
        fileDate = re.findall('\d\d-\w\w\w-\d\d\d\d', fileName)
        expTime = re.findall('-(\d\d\.\d\d\.\d\d)-', fileName)
        expName = re.findall('-M\w\w\d-(.*)\.', fileName)


        # Checking the format.
        if len(fileDate) == 0 or len(expTime) == 0 or len(expName) == 0:
            print('ERROR: Bad file format: %s' % fileName)
            return

        if (workedFiles % WAIT_INTERVAL == 0 and workedFiles != 0):
            ftp.quit()
            print('Sleeping..')
            time.sleep(WAIT_HOURS * 60 * 60)
            ftp = FTP('132.64.59.3')
            ftp.login(userName, pwd)


        fileDate = fileDate[0]
        expTime = expTime[0]
        expName = expName[0]

        # Output parent dir
        outputParentDir = path.join(localPath, fileDate) + "_Chris"

        if not path.isdir(outputParentDir):
            mkdir(outputParentDir)

        # Creating the localdir
        outputLocalDir = path.join(outputParentDir, expName + "_" + expTime)
        if os.path.exists(outputLocalDir):
            print('Directory exists. continuing.')
            continue

        mkdir(outputLocalDir)

        try:
            # Creating the final destination of the file.
            outputFile = path.join(outputLocalDir, fileName)

            if os.path.exists(outputFile):
                print('File Exists. Continuing.')
                continue

            with open(outputFile,'wb') as fileHandle:
                print('Start retrieving.')
                retTime = time.time()
                ftp.retrbinary('RETR %s' % fileName, fileHandle.write)
                retTime = time.time() - retTime
                print('Done retrieving. in %d minutes' % (retTime / 60))

            #conduct(outputLocalDir)
            os.system('sbatch --mem=64g --gres gpu:m60:1 -c4 --time=0-12 ./processVideo.bash %s' % outputLocalDir)
        except Exception as exp:
            print(exp)
            print ('Error. Continuing to the next file.')
            os.rmdir(outputLocalDir)

        workedFiles += 1















if __name__ ==  "__main__":
    main()







