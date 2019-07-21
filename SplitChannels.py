import sys
import cv2

from skvideo.io import FFmpegWriter
from time import time, sleep

from os import path


def splitChannel(params):
    videoFileName = params[0]
    wantedChannel = params[1]

    # Getting the file path
    inputPath = path.dirname(videoFileName)
    baseName = ".".join(path.basename(videoFileName).split(".")[0:-1])
    # extension = path.basename(videoFileName).split(".")[1]
    extension = 'mp4'

    # Preparing out file.
    channelFileUncompressed = path.join(inputPath, baseName + "_" + str(wantedChannel) + "_Full." + extension)
    channelFileCompressed = path.join(inputPath, baseName + "_" + str(wantedChannel) + "_Compressed." + extension)

    print('Writing into: ' + channelFileUncompressed)
    print('Writing into: ' + channelFileCompressed)

    # OpenCV loading.
    cap = cv2.VideoCapture(videoFileName)

    if (not cap.isOpened):
        print("Couldn't open the file.")
        return

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameNumber = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('fourcc: ' + str(fourcc) + ', fps: ' + str(fps) + ', dims: ' + str((height, width)) + ', Frames: ' + str(
        frameNumber))

    videoWriterUncomprseed = FFmpegWriter(channelFileUncompressed, outputdict={'-crf': '0' })
    videoWriterComprseed = FFmpegWriter(channelFileCompressed, outputdict={'-crf': '25' })


    for frameNum in range(frameNumber):
    #for frameNum in range(2000):
        rTime = time()
        success, readFrame = cap.read()
        rElapsed = time() - rTime

        if success:
            channel = readFrame[:, :, wantedChannel]
            wTime = time()

            videoWriterUncomprseed.writeFrame(channel)
            videoWriterComprseed.writeFrame(channel)

            wElapsed = time() - wTime

        print(str(frameNum) + '. Written frame:' + str(frameNum) + ". Read Time: " + str(rElapsed) + ". Write Time: " + str(
            wElapsed))

        #sleep(0.01)

    videoWriterComprseed.close()
    videoWriterUncomprseed.close()


def main():
    # Wrong usage
    if (len(sys.argv) == 1):
        print("SplitChannels.py <videoFile>")
        return

    videoFileName = sys.argv[1]
    print('Splitting ' + sys.argv[1] + ".")

    splitChannel((videoFileName, 0))


    #with Pool(processes=3) as p:
        #p.map(splitChannel, list(zip((videoFileName,) * 3, range(3))))
        #p.map(splitChannel, (videoFileName,0))


if __name__ == "__main__":
    main()
