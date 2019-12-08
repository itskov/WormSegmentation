import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from skvideo.io import FFmpegWriter


class AngleVisualizer:
    def __init__(self, exp, tracks):
        self._exp = exp
        self._tracks = tracks


    def visualize(self):

        writer = FFmpegWriter('/home/itskov/Temp/out.mp4', outputdict={'-crf': '15'})

        if 'chemPos' not in self._exp._pointsOfInterest or \
                'chemRad' not in self._exp._pointsOfInterest:
            print('Error: cant find chem point.')
            return

        # First calculate the interesting frames
        allFrames = np.concatenate([tuple(t._trackFrames) for t in self._tracks])
        relevantFrames = range(np.min(allFrames), np.max(allFrames))

        cap = self._exp._cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, np.min(allFrames))

        for i,frame in enumerate(relevantFrames):
            _, currentFrame = cap.read()
            curIm = Image.fromarray(currentFrame).convert('RGB')

            for t in self._tracks:
                if frame in t._trackFrames:
                    currentPos = np.fliplr(t.getPos(frame))
                    currentStep = np.fliplr(t.getStep(frame))

                    if (currentStep[0,0] is None or currentStep[0,1] is None):
                        continue

                    stepsNorm = np.linalg.norm(currentStep)

                    if (stepsNorm != 0):
                        curImDraw = ImageDraw.Draw(curIm)

                        # Calculate line end point.
                        endPoint = currentPos + (currentStep / stepsNorm) * 100
                        curImDraw.line((currentPos[0, 0], currentPos[0, 1], endPoint[0, 0], endPoint[0, 1]), fill='red' , width=3)
            writer.writeFrame(np.asarray(curIm).copy())
            print('Visualizing frame: ' + str(i))

        writer.close()


















