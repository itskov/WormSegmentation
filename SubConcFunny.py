import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skvideo.io import FFmpegWriter

cap = cv2.VideoCapture('/home/itskov/workspace/lab/DeepSemantic/WormSegmentation/The Monkey Business Illusion-IGQmdoK_ZfY.mp4')
movieLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

font = ImageFont.truetype("FreeSans.ttf", 120)

videoWriterRaw = FFmpegWriter('/home/itskov/Temp/Test.mp4', outputdict={'-crf': '20'})

l = 4
for i in range(movieLength):
    _, readFrame = cap.read()

    if i > 300:
        if i % 125 == 0:
            curImg = Image.fromarray(readFrame)
            curImRawDraw = ImageDraw.Draw(curImg)
            curImRawDraw.text((150,150),'JOIN ME!', (0, 0, 255), font=font)
            readFrame = np.asarray(curImg).copy()
            for j in range(l):
                videoWriterRaw.writeFrame(np.asarray(readFrame).copy())
        else:
            videoWriterRaw.writeFrame(np.asarray(readFrame).copy())

        print('Frame: %d' % (i,))


cap.release()
videoWriterRaw.close()